# Autoresearch Extension: `autoexperiment`

## Overview
This extends Karpathy's autoresearch pattern by requiring the agent to maintain a causal model (`theory.md`) and track change-type effectiveness (`credit.json`) alongside optimizing the artifact.

## Core Loop (standard autoresearch + theory maintenance)

### Before each experiment:
1. Read `theory.md` and `credit.json`
2. Write a falsifiable hypothesis to `theory.md` under "Active Hypothesis"
   - Format: "I believe X because Y; changing Z should improve by ~N%"
3. Pick the next change to try, biased by `credit.json` scores but with occasional exploration

### After each experiment:
1. Append results to `theory.md` (confirmed/disconfirmed + actual delta)
2. Update `credit.json` (increment attempts, recalculate avg_delta)
3. Run evaluation and keep/revert based on val_bpb improvement

## Change Categories
When modifying `train.py`, categorize your change as one of:
- `norm_layer_changes`: LayerNorm, RMSNorm, BatchNorm variants
- `optimizer_changes`: Adam, SGD, Muon, etc.
- `attention_pattern`: Attention mechanism changes
- `architecture_changes`: Width, depth, embedding size
- `regularization_changes`: Dropout, weight decay, etc.

## Hypothesis Writing Guidelines
- Be specific: name the exact mechanism you expect to improve
- Be falsifiable: if wrong, the results should clearly contradict the hypothesis
- Estimate expected improvement: "~0.5%" or "~2x faster convergence"
- Reference past experiments: "Based on exp12, X seems to help; I believe Y is the mechanism"

## Credit Score Algorithm
Sort change types by: `avg_delta * success_rate`
- Explore low-credit types occasionally (at least 1 in 5 experiments)
- High credit types get priority but don't monopolize

## Artifact Outputs
- `theory.md`: Human-readable causal model, transferable to new projects
- `credit.json`: Machine-readable exploration map for future runs

## Commands
```bash
uv sync
uv run prepare.py
uv run train.py  # run one experiment
```

## Success Criteria
- val_bpc decreases over experiments
- `theory.md` grows with confirmed/disconfirmed hypotheses
- `credit.json` shows clear signal in change-type effectiveness
