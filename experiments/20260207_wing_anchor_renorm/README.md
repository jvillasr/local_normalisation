# 20260207_wing_anchor_renorm

## Objective
Evaluate a new Balmer wing-anchor renormalisation against current local normalisation baselines, using the FWHM-mask pilot set.

## Pilot data
- Spectra: 12 pilot stars (from existing FWHM-mask pilot figure set)
- Lines evaluated: HDELTA, HGAMMA, HBETA, HALPHA
- Baseline reference: full normalisation (`boss_v6_2_1_normalised`) and Lorentzian+continuum fit diagnostics

## Methods compared
- `p98_k1p5_fit`: p98 renorm, Balmer mask `k=1.5`, stage `fit`
- `p98_k2_fit`: p98 renorm, Balmer mask `k=2.0`, stage `fit`
- `none_k1p5_fit`: no renorm, Balmer mask `k=1.5`, stage `fit`
- `wing_k1p5_fit`: wing-anchor renorm, Balmer mask `k=1.5`, stage `fit`
- `wing_k2_fit`: wing-anchor renorm, Balmer mask `k=2.0`, stage `fit`

## Commands used
Main pilot run:

```bash
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_wing_anchor_renorm/pilot_eval_v1
```

Parameter scans (no per-star plots):

```bash
# scan_a
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_wing_anchor_renorm/scan_a --no-make-star-plots \
  --wing-inner-scale 0.8 --wing-outer-scale 2.5 --wing-anchor-percentile 95 \
  --wing-clip-min 0.75 --wing-clip-max 1.35

# scan_b
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_wing_anchor_renorm/scan_b --no-make-star-plots \
  --wing-inner-scale 1.0 --wing-outer-scale 2.0 --wing-anchor-percentile 98 \
  --wing-clip-min 0.85 --wing-clip-max 1.20

# scan_c
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_wing_anchor_renorm/scan_c --no-make-star-plots \
  --wing-inner-scale 1.2 --wing-outer-scale 2.8 --wing-anchor-percentile 97 \
  --wing-clip-min 0.85 --wing-clip-max 1.20

# scan_d
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_wing_anchor_renorm/scan_d --no-make-star-plots \
  --wing-inner-scale 1.5 --wing-outer-scale 3.0 --wing-anchor-percentile 98 \
  --wing-clip-min 0.90 --wing-clip-max 1.15
```

## Quantitative outcomes
From `pilot_eval_v1/summary_by_method.csv` (lower composite is better):

- `p98_k1p5_fit`: `0.572458` (best)
- `none_k1p5_fit`: `0.584804`
- `wing_k1p5_fit`: `0.606771`
- `p98_k2_fit`: `0.624273`
- `wing_k2_fit`: `0.780728`

Additional checks:
- `wing_k1p5_fit` vs `p98_k1p5_fit` on `abs_bias_local`: improved `4/48`, worse `44/48`.
- Signed bias remains mostly negative:
  - `p98_k1p5_fit` median `bias_local = -0.0455`
  - `wing_k1p5_fit` median `bias_local = -0.0545`

Parameter scans did not change ranking: `p98_k1p5_fit` remained best in all scans.

## Artefacts
Primary outputs:
- `pilot_eval_v1/metrics_by_line.csv`
- `pilot_eval_v1/summary_by_method.csv`
- `pilot_eval_v1/summary_by_method_line.csv`
- `pilot_eval_v1/method_metric_boxplots.png`
- `pilot_eval_v1/method_composite_scores.png`
- `pilot_eval_v1/plots/<method>/*_balmer_local_vs_full_*.png`
- `pilot_eval_v1/plots/<method>/*_balmer_lorentzian_fits_*.png`

Scan outputs:
- `scan_a/`, `scan_b/`, `scan_c/`, `scan_d/`
- Logs: `scan_a.log`, `scan_b.log`, `scan_c.log`, `scan_d.log`

## Decision
Current status: **NO-GO for whole-sample rollout**.

Reason:
- Wing-anchor v1 does not outperform the current best local baseline (`p98_k1p5_fit`) on bias/kink composite metrics.
- Broad-line continuum underestimation remains substantial.

## Next step
Implement a second method variant that explicitly targets negative continuum bias (for example upper-envelope/asymmetric continuum fitting) and re-run this exact evaluator for fair comparison.
