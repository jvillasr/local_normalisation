# 2026-03-31 Parallel Integration Plan

This note records the branch sequence chosen after merging the Balmer-emission
fix into `main`.

## Decision
- Merge `fix/halpha-emission-p98` into `main` first.
- Create a fresh pilot branch from updated `main`.
- Cherry-pick the field-parallel worker commit from
  `exp/20260330_field_parallel` onto that new branch.
- Validate the combined branch with the 32-worker pilot before merging any
  parallel execution changes to `main`.

## Why this order
- Both the emission fix and the field-parallel work diverged directly from `main`.
- Both branches modify `scripts/local_normalise_boss_spectra.py`.
- The emission fix changes method behaviour, while the parallel branch changes run
  orchestration and registry handling.
- The pilot should test the final method logic together with the parallel
  execution path, not the older pre-fix method state.

## Integration branch
- branch name: `exp/20260331_emission_parallel_pilot`
- base: `main` after the `fix/halpha-emission-p98` merge
- applied parallel commit: `e835780cbf8aabffe8085a33e5589a10510fffb2`

## Launch rule
- Use the shared SDSS `uv` environment from the coordinating repository, not the
  local `local_normalisation/.venv`, when the run must write
  `processed_registry.parquet`.

## Pilot scope
- manifest: 1000 spectra across 1000 fields
- workers: 32
- method flags:
  - `--fit-stage balmer_subtract`
  - `--p98-renorm`
  - `--renorm-percentile 98`
  - `--balmer-mask-k 1.5`
  - `--balmer-mask-stage fit`
