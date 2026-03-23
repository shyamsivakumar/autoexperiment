"""
prepare.py - fixed data/tokenizer/eval setup (same as autoresearch)
"""
import os
import urllib.request
import torch
import numpy as np

DATA_DIR = "/tmp"
DATA_FILE_TXT = os.path.join(DATA_DIR, "train.txt")
DATA_FILE = os.path.join(DATA_DIR, "train.bin")
DATA_FILE_VAL = os.path.join(DATA_DIR, "val.bin")

B = 4
T = 128
val_bpc_target = None

data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return np.array([self.stoi[c] for c in text], dtype=np.int32)
    
    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])

def download_data():
    if not os.path.exists(DATA_FILE_TXT):
        print(f"Downloading {data_url}...")
        urllib.request.urlretrieve(data_url, DATA_FILE_TXT)
        print(f"Saved to {DATA_FILE_TXT}")

def get_data():
    download_data()
    
    if not os.path.exists(DATA_FILE):
        print("Tokenizing data...")
        with open(DATA_FILE_TXT, 'r') as f:
            text = f.read()
        
        tokenizer = CharTokenizer(text)
        tokens = tokenizer.encode(text)
        
        # Split 90/10 for train/val
        n = int(len(tokens) * 0.9)
        train_tokens = tokens[:n]
        val_tokens = tokens[n:]
        
        train_tokens.astype(np.int32).tofile(DATA_FILE)
        val_tokens.astype(np.int32).tofile(DATA_FILE_VAL)
        
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Train tokens: {len(train_tokens)}, Val tokens: {len(val_tokens)}")
    
    train = np.fromfile(DATA_FILE, dtype=np.int32)
    val = np.fromfile(DATA_FILE_VAL, dtype=np.int32)
    return train, val

def get_vocab_size():
    get_data()  # ensure data exists
    with open(DATA_FILE_TXT, 'r') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    return tokenizer.vocab_size

def get_batch(split, device):
    train, val = get_data()
    data = train if split == "train" else val
    ix = torch.randint(len(data) - T, (B,))
    x = torch.stack([torch.from_numpy(data[i:i+T].copy()) for i in ix]).long()
    y = torch.stack([torch.from_numpy(data[i+1:i+T+1].copy()) for i in ix]).long()
    return x.to(device), y.to(device)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calc_bpc(model):
    torch.set_grad_enabled(False)
    model.eval()
    losses = []
    for _ in range(10):
        x, y = get_batch('val', device)
        _, loss = model(x, y)
        losses.append(loss.item())
    bpc = np.mean(losses) / np.log(2)
    model.train()
    torch.set_grad_enabled(True)
    return bpc

def eval_model(model):
    return calc_bpc(model)

if __name__ == "__main__":
    download_data()
    get_data()
    print("Data ready.")
