# AGENTS.md — local_normalisation

## Project intent
Build and maintain the local-normalisation pipeline that prepares windowed BOSS spectra as SpecFANN inputs, with demonstrably better Balmer-region continuum fidelity than the full-spectrum normalisation baseline.

Outputs feed `../physparams_paper` (production local-normalised datasets) and `../multiplicity_paper` (via shared datasets in the central storage root).

## Repository layout
- `scripts/`: runnable entry points and diagnostics.
  - `local_normalise_boss_spectra.py` — main production script
  - `fit_stage_methods.py` — Balmer fit-stage plugin methods
  - `evaluate_balmer_renorm_methods.py` — quantitative method comparison
  - `plot_balmer_local_vs_full.py` — diagnostic overlay plots
- `src/local_norm/`: pipeline modules (expanding from script-first code).
- `configs/`: named parameter sets for reproducible runs (`baseline_full.yml`, `local_window_v1.yml`).
- `experiments/`: dated pilots and exploratory outputs (`<YYYYMMDD>_<name>/`).
- `reports/`: quantitative comparison tables and analysis notes.
- `docs/`: design notes, decisions, and workflow guidance.
- `tests/`: unit and regression tests.

## Canonical data pointers
- Shared production output root (local-normalised windowed datasets):
  `/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_local_norm/`
- Full-spectrum normalised reference (authoritative baseline for comparison):
  `/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_normalised/`
- Shared final sample (locked at 2026-03-18 per `../physparams_paper/DECISIONS.md`):
  `/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_cache/spAll-v6_2_1_mwm_ob_observed.parquet`

## Environment and execution policy
- Use the shared SDSS `uv` environment.
- Run Python as `uv run python ...` or `uv run python -m <module>`. Do not use system Python.
- UV cache: `UV_CACHE_DIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/uv-cache`
- HotStarsBOSS normalisation code (external dependency, do not edit):
  `/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/HotStarsBOSS/code`

## Commit and push policy
Validated work is not complete until it is committed and pushed, unless explicitly stated otherwise. Never leave finished changes uncommitted or unpushed. This is especially mandatory for production script changes, config updates, and any change that a downstream workflow or dataset depends on. If a push is rejected, fetch and rebase safely, then push again. Do not stop at a dirty branch.

## Branch workflow
One branch per methodological hypothesis. See `docs/branch_workflow.md` for naming conventions (`feat/`, `exp/`, `fix/`). Minimum merge evidence: quantitative comparison against baseline in `reports/`, pilot plots in `experiments/`, relevant tests passing, failure modes noted.

## Method status and promotion
Use three explicit states: `experimental`, `optional`, `default`. A method is only canonical when `../physparams_paper/DECISIONS.md` names it and the production command explicitly passes any required non-default flags. Do not treat experiment READMEs or branch names as promotion.

**Current production default:** `--fit-stage balmer_subtract` (`bsub_p98`), confirmed in `../physparams_paper/DECISIONS.md` entry 2026-03-30.

## Production run requirements
Before any production or paper-facing run, record in `../physparams_paper/NOTES.md`:
- branch name and HEAD commit hash
- `git status --short` output (must be clean, or dirty state explicitly approved)
- exact command used (including all non-default flags)

## Worktree note
This directory (`local_normalisation`) is a linked worktree. The main git repo lives at `../local_normalised_data`. This is transparent for daily use. Do not `rm -rf ../local_normalised_data` without migrating the git history first.

## File safety policy
Always ask for explicit confirmation before removing files, overwriting outputs, or running scripts that write to the shared central storage root.
