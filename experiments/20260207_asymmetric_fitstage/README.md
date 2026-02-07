# 20260207_asymmetric_fitstage

## Objective
Test genuinely different **fit-stage** continuum approaches (not post-fit renorm) that target the core problem: systematic continuum underestimation in broad Balmer wings caused by the sigma-clip spline being pulled down by wing absorption.

## Branch
`claude/20260207_asymmetric_fitstage` (branched from `exp/20260207_wing_anchor_renorm` @ `3ca6d15`)

## Background
Previous wing-anchor and upper-envelope **post-fit** renorm methods on branch `exp/20260207_wing_anchor_renorm` failed to beat the `p98_k1p5_fit` baseline (composite = 0.572). The fundamental issue is that post-fit corrections cannot fully recover lost continuum shape information after the sigma-clip spline has already fitted through the broad wings.

## New fit-stage methods implemented
Three alternative continuum estimators that replace the sigma-clip spline for Balmer windows:

1. **`quantile_spline`** — IRLS quantile regression spline targeting the tau-th quantile (default tau=0.85). Naturally biases the fit upward toward the pseudo-continuum, avoiding wing pull-down.

2. **`penalised_upper`** — Iterative upper-envelope spline. Repeatedly removes points below the current fit and refits, converging on the upper boundary of the flux distribution.

3. **`lorentzian_deblend`** — Joint Lorentzian absorption + quadratic polynomial continuum fit. Physically motivated: the Lorentzian profile absorbs the wing flux while the polynomial captures the true pseudo-continuum shape.

Each can be combined with the existing p98 renorm as a fine-tuning step.

## Implementation
- New module: `scripts/fit_stage_methods.py`
- Extended `scripts/evaluate_balmer_renorm_methods.py`:
  - `MethodConfig.fit_mode` field (standard | quantile_spline | penalised_upper | lorentzian_deblend)
  - Method spec format extended to 5 colon-separated fields: `name:renorm_mode:mask_k:mask_stage:fit_mode`
  - CLI arguments for quantile spline (`--qs-tau`, `--qs-knot-spacing`, etc.) and penalised upper (`--pu-*`)

## Staged testing funnel

### Stage 1: Single-star triage (hardest case)
Star: `spec-017017-59626-27021597913517517` (worst abs_bias = 0.067 under baseline)

| Method | Composite |
|---|---:|
| qs85_k1p5_fit | **0.313** |
| lordeblend_k1p5_fit | 0.568 |
| pupper_k1p5_fit | 0.573 |
| p98_k1p5_fit (baseline) | 0.698 |

**Result:** Quantile spline cuts composite by 55% on the hardest case. All three methods pass triage.

### Stage 1b: Tau parameter scan
Tested tau ∈ {0.75, 0.80, 0.85, 0.90, 0.95} with/without p98 renorm on the hard case.

| Tau | qs_none composite | qs+p98 composite |
|-----|------------------:|-----------------:|
| 0.75 | 0.580 | 0.309 |
| 0.80 | 0.567 | 0.312 |
| 0.85 | 0.313 | 0.313 |
| 0.90 | 0.313 | 0.313 |
| 0.95 | 0.286 | 0.307 |

**Result:** qs+p98 is remarkably stable across tau; tau=0.85 selected as default.

### Stage 2: 3-star stress test
Stars: hardest + second-hardest + easiest

| Method | Composite | abs_bias | under_frac |
|---|---:|---:|---:|
| qs85_p98 | **0.035** | 0.021 | 0.00 |
| qs85_none | 0.280 | 0.015 | 0.50 |
| pupper_none | 0.536 | 0.019 | 1.00 |
| lordeblend_none | 0.558 | 0.039 | 1.00 |
| p98_k1p5_fit | 0.672 | 0.062 | 1.00 |

**Result:** qs85_p98 achieves 95% improvement over baseline on the stress set.

### Stage 3: Full 12-star pilot
All 12 pilot stars, with star plots.

| Method | Composite | abs_bias | under_frac | kink_excess |
|---|---:|---:|---:|---:|
| lordeblend_p98 | **0.107** | 0.013 | 0.18 | 0.016 |
| qs85_p98 | 0.274 | 0.020 | 0.50 | 0.017 |
| lordeblend_none | 0.515 | 0.012 | 1.00 | 0.015 |
| qs85_none | 0.520 | 0.016 | 1.00 | 0.017 |
| pupper_p98 | 0.513 | 0.019 | 0.98 | 0.021 |
| p98_k1p5_fit | 0.572 | 0.049 | 1.00 | 0.119 |

**Pairwise vs baseline:**
- `lordeblend_p98` beats baseline on ~83% of line-cases
- `qs85_p98` beats baseline on 79% of line-cases
- All new methods beat baseline on >79% of line-cases

## Recommended method
**`lordeblend_p98`** — Lorentzian deblend (fit_mode) + p98 renorm

- 81% composite improvement (0.572 → 0.107)
- 75% abs_bias improvement (0.049 → 0.013)
- under_frac drops from 1.00 to 0.18
- 87% kink_excess improvement (0.119 → 0.016)
- Zero failure rate

**Rationale:** The Lorentzian profile model is physically motivated — it correctly captures the broad Balmer wing shape, preventing the continuum fit from being pulled down. The p98 renorm provides a final level correction.

**Runner-up:** `qs85_p98` (quantile spline tau=0.85 + p98) — 52% improvement, more robust to non-Balmer-like profiles, useful as a fallback.

## Go/no-go for scale-up
**GO** — `lordeblend_p98` and `qs85_p98` both exceed the scale-up gate criteria:
- Composite beats baseline: YES (0.107 and 0.274 vs 0.572)
- Beats baseline on most Balmer lines: YES (all 4 lines improved)
- No increase in failure rate: YES (0% for both)

Next steps:
1. Integrate `lordeblend_p98` into the production `local_normalise_boss_spectra.py` pipeline
2. Run on the full field set
3. Re-evaluate multiplicity signal post local normalisation

## Reproducibility

```bash
# Stage 1: triage
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_asymmetric_fitstage/triage_hard_case \
  --spec-names spec-017017-59626-27021597913517517 \
  --no-make-star-plots \
  --methods p98_k1p5_fit:p98:1.5:fit:standard qs85_k1p5_fit:none:1.5:fit:quantile_spline pupper_k1p5_fit:none:1.5:fit:penalised_upper lordeblend_k1p5_fit:none:1.5:fit:lorentzian_deblend \
  --qs-tau 0.85

# Stage 2: stress
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_asymmetric_fitstage/stress_3star \
  --spec-names spec-017017-59626-27021597913517517 spec-017153-59638-27021598086391213 spec-103693-60624-63050394807644754 \
  --no-make-star-plots \
  --methods p98_k1p5_fit:p98:1.5:fit:standard qs85_none:none:1.5:fit:quantile_spline qs85_p98:p98:1.5:fit:quantile_spline lordeblend_none:none:1.5:fit:lorentzian_deblend pupper_none:none:1.5:fit:penalised_upper \
  --qs-tau 0.85

# Stage 3: full pilot
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_asymmetric_fitstage/pilot_12star \
  --make-star-plots \
  --methods p98_k1p5_fit:p98:1.5:fit:standard qs85_none:none:1.5:fit:quantile_spline qs85_p98:p98:1.5:fit:quantile_spline lordeblend_none:none:1.5:fit:lorentzian_deblend pupper_none:none:1.5:fit:penalised_upper \
  --qs-tau 0.85

# Stage 3b: combos (no plots)
UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache uv run python scripts/evaluate_balmer_renorm_methods.py \
  --output-dir experiments/20260207_asymmetric_fitstage/pilot_12star_combos \
  --no-make-star-plots \
  --methods p98_k1p5_fit:p98:1.5:fit:standard qs85_p98:p98:1.5:fit:quantile_spline lordeblend_p98:p98:1.5:fit:lorentzian_deblend pupper_p98:p98:1.5:fit:penalised_upper \
  --qs-tau 0.85
```
