# Fast screening workflow

Use a staged funnel to reduce iteration time.

## Stage 1: one-star triage (fast)
Goal: reject clearly bad ideas quickly.

- Pick one hard case (broad/noisy Balmer profile).
- Run `evaluate_balmer_renorm_methods.py` with:
  - `--spec-names <spec-...>`
  - `--no-make-star-plots`
- Compare candidate methods against `p98_k1p5_fit`.

Pass criterion to continue:
- Better or equal `composite_score` than `p98_k1p5_fit` on the test star.

## Stage 2: stress subset (3 stars)
Goal: check transfer beyond one case.

- Use three stars: one broad, one moderate, one pathological/noisy.
- Keep `--no-make-star-plots` for speed.

Pass criterion to continue:
- Candidate beats `p98_k1p5_fit` on median `abs_bias_local` and does not worsen median kink.

## Stage 3: pilot set (12 stars)
Goal: final gate before scale-up.

- Run full pilot with plots.
- Use fixed metrics and ranking tables already produced by the evaluator.

Scale-up gate:
- Candidate must beat `p98_k1p5_fit` on overall composite and on most Balmer lines.
