# autoexperiment

Hypothesis-driven autoresearch loop with causal model output.

## What it does

Extends Karpathy's autoresearch pattern by requiring the agent to maintain a causal model (`theory.md`) and track change-type effectiveness (`credit.json`). Where standard autoresearch finds good solutions without understanding *why*, `autoexperiment` builds transferable knowledge.

## Novel Output Artifact

**`theory.md`** - A human-readable causal model that transfers to new projects. Point a future run at a related problem with this as context and skip weeks of dead ends.

**`credit.json`** - Machine-readable exploration map showing which change types actually work in this domain.

## Comparison vs Standard Autoresearch

| Feature | Autoresearch | autoexperiment |
|---------|--------------|----------------|
| Artifact | `train.py` | `train.py` + `theory.md` + `credit.json` |
| Change selection | Random or heuristics | Credit-weighted + exploration |
| Hypothesis tracking | None | Mandatory before each experiment |
| Knowledge output | Implicit in best model | Explicit in theory.md |
| Setup | 3 files | 5 files |

## Quick Start

```bash
uv sync
uv run prepare.py  # downloads data
git init && git add -A && git commit -m "baseline"
uv run train.py    # run first experiment
```

## Loop Flow

1. Read `theory.md` + `credit.json`
2. Write falsifiable hypothesis
3. Make one change to `train.py`
4. Run experiment, evaluate val_bpc
5. Update `theory.md` (confirm/disconfirm) and `credit.json`
6. Keep change if improved, else revert
7. Repeat

## Community Theories

After a run, paste your `theory.md` to a GitHub Gist and PR the link to `THEORIES.md` under your domain. Before starting a new project, read 2-3 relevant theories for a head start.

## Based on

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) - The base pattern this extends.

---

*MIT License*
