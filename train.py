"""
train.py - agent modifies this file
Starting point: minimal nanoGPT variant on FineWeb
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# hyperparams - agent can modify these
B, T = 4, 128
vocab_size = None  # filled from prepare
max_lr = 1e-3
warmup_iters = 100
min_lr = 5e-5
log_interval = 10
max_iters = 100
weight_decay = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# model components

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.get('n_head', 8)
        self.n_embd = config.get('n_embd', 256)
        self.head_size = self.n_embd // self.n_head
        self.attn_dropout = nn.Dropout(config.get('attn_dropout', 0.0))
        
        # key, query, value projections
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        self.register_buffer("mask", mask)
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.get('n_embd', 256))
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.get('n_embd', 256))
        self.mlp = nn.Sequential(
            nn.Linear(config.get('n_embd', 256), 4 * config.get('n_embd', 256)),
            nn.GELU(),
            nn.Linear(4 * config.get('n_embd', 256), config.get('n_embd', 256)),
            nn.Dropout(config.get('mlp_dropout', 0.0)),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, config.get('n_embd', 256)),
            'wpe': nn.Embedding(T, config.get('n_embd', 256)),
            'drop': nn.Dropout(config.get('embd_dropout', 0.0)),
            'h': nn.ModuleList([Block(config) for _ in range(config.get('n_layer', 8))]),
            'ln_f': nn.LayerNorm(config.get('n_embd', 256)),
        })
        self.lm_head = nn.Linear(config.get('n_embd', 256), vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(self, x, targets=None):
        x = self.transformer.wte(x) + self.transformer.wpe(torch.arange(x.size(1), device=x.device))
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], learning_rate

# -----------------------------------------------------------------------------
# training loop

def get_lr(it):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    return max_lr * (max_iters - it) / (max_iters - warmup_iters)

def train():
    global vocab_size
    from prepare import get_batch, get_vocab_size
    vocab_size = get_vocab_size()
    
    config = {
        'n_embd': 256,
        'n_head': 8,
        'n_layer': 8,
        'attn_dropout': 0.1,  # Added dropout
        'mlp_dropout': 0.1,
        'embd_dropout': 0.0,
    }
    
    model = GPT(config).to(device)
    params, lr = model.configure_optimizers(weight_decay, max_lr)
    optimizer = torch.optim.AdamW(params, lr=lr, fused=device == 'cuda')
    
    print(f"Training on {device}")
    print(f"Config: {config}")
    
    for iter in range(max_iters):
        lr = get_lr(iter)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            
        x, y = get_batch('train', device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if iter % log_interval == 0:
            print(f"iter {iter}: loss {loss.item():.4f}")
    
    # eval
    model.eval()
    with torch.no_grad():
        x, y = get_batch('val', device)
        _, val_loss = model(x, y)
    val_bpc = val_loss.item() / np.log(2)
    print(f"Final val_bpc: {val_bpc:.6f}")
    
    # save
    torch.save(model.state_dict(), '/tmp/gpt.pt')
    return val_bpc

if __name__ == "__main__":
    train()
