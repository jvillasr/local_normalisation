# 2026-02-07 Wing-Anchor Pilot Summary

This note summarises the first wing-anchor renormalisation pilot on branch `exp/20260207_wing_anchor_renorm`.

## Summary judgement
- Best method in this round: `p98_k1p5_fit`.
- Wing-anchor v1 improved over no-renorm but did not beat p98.
- Not sufficient yet for large-scale local normalisation.

## Key numbers
Using `experiments/20260207_wing_anchor_renorm/pilot_eval_v1/summary_by_method.csv`:

| method | composite_score |
|---|---:|
| p98_k1p5_fit | 0.572458 |
| none_k1p5_fit | 0.584804 |
| wing_k1p5_fit | 0.606771 |
| p98_k2_fit | 0.624273 |
| wing_k2_fit | 0.780728 |

Pairwise result:
- `wing_k1p5_fit` vs `p98_k1p5_fit` on `abs_bias_local`: better in 4/48 line-cases, worse in 44/48.

Bias sign check:
- `p98_k1p5_fit` median `bias_local = -0.0455`
- `wing_k1p5_fit` median `bias_local = -0.0545`

Interpretation: local continua remain systematically low vs Lorentzian continuum in the Balmer wings.

## Reproducibility
Main run:

```bash
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_wing_anchor_renorm/pilot_eval_v1
```

Scan runs are documented in:
- `experiments/20260207_wing_anchor_renorm/README.md`

## Follow-up
Prioritise an upper-envelope/asymmetric continuum method and rerun the same evaluator to maintain direct comparability.

## Update: upper-envelope method trial

A second candidate (`upper_envelope`) was implemented and evaluated with the same pilot framework.

Main run:

```bash
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_wing_anchor_renorm/pilot_eval_upper_v1 \
  --methods \
    p98_k1p5_fit:p98:1.5:fit \
    p98_k2_fit:p98:2.0:fit \
    upper_k1p5_fit:upper_envelope:1.5:fit \
    upper_k2_fit:upper_envelope:2.0:fit
```

Results (`summary_by_method.csv`):
- `p98_k1p5_fit`: `0.572458`
- `upper_k1p5_fit`: `0.607025`
- `p98_k2_fit`: `0.624273`
- `upper_k2_fit`: `0.746034`

Tuning scan of upper-envelope parameters improved `upper_k1p5_fit` only to `0.594138` (best), still above `0.572458`.

Updated judgement: upper-envelope v1/v1-tuned is still not sufficient to replace p98 baseline.
