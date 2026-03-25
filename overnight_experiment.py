#!/usr/bin/env python3
"""
Overnight experiment runner for autoexperiment
Runs multiple experiments, tracks results, updates theory.md and credit.json
"""
import subprocess
import json
import os
import re
import time
from datetime import datetime

EXPERIMENTS_DIR = "/tmp/autoexperiment_runs"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def run_experiment(exp_num, change_desc, config_change_fn=None):
    """Run one experiment and return the val_bpc result."""
    print(f"\n{'='*60}")
    print(f"Experiment {exp_num}: {change_desc}")
    print(f"{'='*60}")
    
    # Backup current train.py
    with open('train.py', 'r') as f:
        original_train = f.read()
    
    # Apply config change if provided
    if config_change_fn:
        config_change_fn()
    
    # Run training
    result = subprocess.run(['uv', 'run', 'train.py'], 
                          capture_output=True, text=True)
    
    # Extract final val_bpc
    val_bpc = None
    for line in result.stdout.split('\n'):
        if 'Final val_bpc' in line:
            match = re.search(r'val_bpc:\s*([\d.]+)', line)
            if match:
                val_bpc = float(match.group(1))
    
    # Restore original train.py (keep config changes if successful)
    # Actually for this demo, we always restore - agent would decide
    with open('train.py', 'w') as f:
        f.write(original_train)
    
    print(f"Result: val_bpc = {val_bpc}")
    return val_bpc

def update_artifacts(exp_num, change_desc, val_bpc, baseline_bpc, kept):
    """Update theory.md and credit.json with experiment results."""
    
    # Update theory.md
    with open('theory.md', 'r') as f:
        theory = f.read()
    
    delta = baseline_bpc - val_bpc
    result_str = "CONFIRMED" if delta > 0 else "DISCONFIRMED"
    
    new_entry = f"\n- exp{exp_num}: {change_desc} → {result_str} (delta: {delta:+.4f})"
    
    if "## Confirmed" in theory:
        theory = theory.replace("## Confirmed", f"## Confirmed{new_entry}")
    else:
        theory = theory.replace("## Current Understanding", 
                               f"## Current Understanding{new_entry}")
    
    with open('theory.md', 'w') as f:
        f.write(theory)
    
    # Update credit.json
    with open('credit.json', 'r') as f:
        credit = json.load(f)
    
    # Categorize the change
    category = categorize_change(change_desc)
    if category not in credit:
        credit[category] = {"attempts": 0, "successes": 0, "avg_delta": 0, "experiments": []}
    
    credit[category]["attempts"] += 1
    credit[category]["experiments"].append({
        "exp": exp_num,
        "change": change_desc,
        "delta": delta,
        "success": delta > 0
    })
    if delta > 0:
        credit[category]["successes"] += 1
    
    # Recalculate avg_delta
    deltas = [e["delta"] for e in credit[category]["experiments"]]
    credit[category]["avg_delta"] = sum(deltas) / len(deltas)
    
    with open('credit.json', 'w') as f:
        json.dump(credit, f, indent=2)
    
    print(f"Updated theory.md and credit.json")
    print(f"Category: {category}, Delta: {delta:+.4f}, Kept: {kept}")

def categorize_change(change_desc):
    """Categorize the type of change."""
    desc = change_desc.lower()
    if 'lr' in desc or 'learning' in desc:
        return "learning_rate_changes"
    elif 'dropout' in desc or 'regularization' in desc or 'weight_decay' in desc:
        return "regularization_changes"
    elif 'layer' in desc or 'n_embd' in desc or 'n_head' in desc or 'embed' in desc:
        return "architecture_changes"
    elif 'optimizer' in desc or 'adam' in desc or 'sgd' in desc or 'muon' in desc:
        return "optimizer_changes"
    elif 'attention' in desc or 'head' in desc:
        return "attention_pattern"
    else:
        return "other_changes"

def apply_change_and_get_desc(change_num):
    """Apply a change and return description. Returns None when no more changes."""
    
    changes = [
        # (description, function to apply change)
        ("increased dropout to 0.15", lambda: set_config({'attn_dropout': 0.15, 'mlp_dropout': 0.15})),
        ("decreased dropout to 0.05", lambda: set_config({'attn_dropout': 0.05, 'mlp_dropout': 0.05})),
        ("increased n_layer to 10", lambda: set_config({'n_layer': 10})),
        ("decreased n_layer to 6", lambda: set_config({'n_layer': 6})),
        ("increased n_head to 12", lambda: set_config({'n_head': 12})),
        ("decreased n_head to 4", lambda: set_config({'n_head': 4})),
        ("increased n_embd to 320", lambda: set_config({'n_embd': 320})),
        ("decreased n_embd to 192", lambda: set_config({'n_embd': 192})),
        ("increased max_lr to 2e-3", lambda: set_lr(2e-3)),
        ("decreased max_lr to 5e-4", lambda: set_lr(5e-4)),
        ("added weight decay to 0.05", lambda: set_weight_decay(0.05)),
        ("removed weight decay (0.0)", lambda: set_weight_decay(0.0)),
    ]
    
    if change_num >= len(changes):
        return None, None
    
    desc, fn = changes[change_num]
    
    with open('train.py', 'r') as f:
        content = f.read()
    
    fn()
    
    return desc, content

def set_config(updates):
    """Update config values in train.py"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    for key, val in updates.items():
        # Handle float values
        if isinstance(val, float):
            pattern = rf"('{key}':\s*)[0-9.e+-]+"
            content = re.sub(pattern, rf"\g<1>{val}", content)
    
    with open('train.py', 'w') as f:
        f.write(content)

def set_lr(lr):
    with open('train.py', 'r') as f:
        content = f.read()
    content = re.sub(r"(max_lr\s*=\s*)[0-9.e+-]+", rf"\g<1>{lr}", content)
    with open('train.py', 'w') as f:
        f.write(content)

def set_weight_decay(wd):
    with open('train.py', 'r') as f:
        content = f.read()
    content = re.sub(r"(weight_decay\s*=\s*)[0-9.e+-]+", rf"\g<1>{wd}", content)
    with open('train.py', 'w') as f:
        f.write(content)

def get_current_config():
    """Get current config from train.py"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    config = {}
    for key in ['max_lr', 'weight_decay', 'n_layer', 'n_embd', 'n_head', 'attn_dropout', 'mlp_dropout']:
        match = re.search(rf"'{key}':\s*([0-9.e+-]+)", content)
        if match:
            try:
                config[key] = float(match.group(1))
            except:
                config[key] = match.group(1)
    return config

def run():
    print("Starting overnight experiment at", datetime.now())
    
    # Get baseline
    baseline_bpc = run_experiment(0, "baseline (no changes)")
    
    results = [(0, baseline_bpc, "baseline", True)]
    
    # Run experiments
    for i in range(10):  # 10 experiments
        change_desc, original = apply_change_and_get_desc(i)
        if change_desc is None:
            print("No more changes to try")
            break
        
        val_bpc = run_experiment(i+1, change_desc)
        
        # Decide keep/revert
        kept = val_bpc < baseline_bpc
        if kept:
            baseline_bpc = val_bpc  # Update baseline
        else:
            # Restore original
            with open('train.py', 'w') as f:
                f.write(original)
        
        update_artifacts(i+1, change_desc, val_bpc, baseline_bpc, kept)
        results.append((i+1, val_bpc, change_desc, kept))
        
        # Commit
        os.system('git add -A && git commit -m "exp{}: {} (val_bpc={:.4f})" 2>/dev/null')
        
        time.sleep(2)
    
    # Generate progress chart
    generate_chart(results)
    
    print("\n" + "="*60)
    print("Overnight experiment complete!")
    print("="*60)
    print(f"\nResults:")
    for exp, bpc, desc, kept in results:
        status = "KEEP" if kept else "revert"
        print(f"  exp{exp}: {bpc:.4f} ({status}) - {desc}")
    
    print("\nPush results with: git push origin main")

def generate_chart(results):
    """Generate a simple ASCII progress chart"""
    with open('/tmp/progress_chart.txt', 'w') as f:
        f.write("Autoexperiment Progress Chart\n")
        f.write("="*50 + "\n\n")
        
        min_bpc = min(r[1] for r in results)
        max_bpc = max(r[1] for r in results)
        
        for exp, bpc, desc, kept in results:
            bar_len = int((bpc - min_bpc) / (max_bpc - min_bpc + 0.001) * 40)
            bar = "█" * (40 - bar_len) + "░" * bar_len
            status = "✓" if kept else "✗"
            f.write(f"exp{exp:2d}: {bar} {bpc:.4f} {status}\n")
            f.write(f"       {desc}\n\n")
    
    print(f"\nProgress chart saved to /tmp/progress_chart.txt")

if __name__ == "__main__":
    run()
