# local_normalised_data

Local BOSS normalisation project workspace for SpecFANN input preparation.

## Aim
Build a local normalisation pipeline that is demonstrably better than the full-spectrum normalisation in:
`HotStarsBOSS/code/boss_normalisation.py`

with the specific goal of improving SpecFANN input quality (especially broad Balmer regions).

## Project layout
- `src/local_norm/`: pipeline modules (to be expanded from script-first code).
- `scripts/`: runnable entry points, pilots, and diagnostics.
- `configs/`: named parameter sets for reproducible runs.
- `tests/`: unit and regression tests.
- `data/products/`: produced HDF5 outputs, registries, and configs.
- `experiments/`: dated pilots and exploratory outputs.
- `reports/`: analysis notes and summary tables.
- `docs/`: design notes, decisions, and workflow guidance.

## Workflow
1. Create a branch per method change, e.g. `feat/wing-anchor-renorm`.
2. Add or update config(s) in `configs/`.
3. Run pilot scripts and save outputs under `experiments/<YYYYMMDD>_<name>/`.
4. Record quantitative comparison against baseline in `reports/`.
5. Merge only changes that improve agreed metrics and pass tests.

## Baseline comparison
Current baseline is the full-spectrum normalisation from HotStarsBOSS. Any local method should be evaluated against this baseline on:
- Balmer continuum bias around broad lines.
- Kink/edge artefacts near window boundaries.
- Stability across noisy and emission-contaminated spectra.

