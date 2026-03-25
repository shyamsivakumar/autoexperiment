# Theory Model

## Current Understanding
_Agent grows this section over experiments. Example:_
- _RMSNorm consistently outperforms BatchNorm in this regime (3/3 tests)_
- _Muon optimizer: strong signal for 2D weights (high credit score)_

## Current Understanding
- Dropout (0.1) improves regularization and slightly reduces val_bpc (+1 experiment confirmed)

## Active Hypothesis
_Trying n_layer increase (8→12) with dropout to see if deeper model benefits from more regularization._

## Disconfirmed
_- exp1: Lower LR (3e-4 vs 1e-3) → WRONG (degraded val_bpc 3.757 → 3.875, +0.12). LR 1e-3 was already near-optimal for this setup._
_- exp2: n_embd 256→384 → WRONG (degraded val_bpc 3.757 → 3.806, +0.05). Larger model underfits in 100 iters._

## Confirmed
- exp10: decreased max_lr to 5e-4 → DISCONFIRMED (delta: -0.0847)
- exp9: increased max_lr to 2e-3 → DISCONFIRMED (delta: -0.1056)
- exp8: decreased n_embd to 192 → DISCONFIRMED (delta: -0.0345)
- exp7: increased n_embd to 320 → DISCONFIRMED (delta: -0.2064)
- exp6: decreased n_head to 4 → DISCONFIRMED (delta: +0.0000)
- exp5: increased n_head to 12 → DISCONFIRMED (delta: -0.1168)
- exp4: decreased n_layer to 6 → DISCONFIRMED (delta: -0.0434)
- exp3: increased n_layer to 10 → DISCONFIRMED (delta: -0.0664)
- exp2: decreased dropout to 0.05 → DISCONFIRMED (delta: -0.0488)
- exp1: increased dropout to 0.15 → DISCONFIRMED (delta: -0.1086)
_- exp3: dropout 0.0→0.1 → CORRECT (improved val_bpc 3.757 → 3.749, -0.008). Regularization helps._
