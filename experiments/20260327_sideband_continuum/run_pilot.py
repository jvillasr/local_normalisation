#!/usr/bin/env python3
"""
Pilot comparison: standard vs bsub vs sideband_linear continuum for Hdelta.

Processes 2 diagnostic stars with each method, generates overlay plots showing
raw flux + all three continua and normalised flux for each.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add scripts to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from local_normalise_boss_spectra import (
    BALMER_CENTRES,
    BALMER_LINES,
    build_groups,
    build_line_windows,
    import_hotstars_normalisation,
    load_spectrum,
    local_normalise_windows,
    renorm_p98_continuum,
    resolve_bundle_path,
    restrict_hdelta_group_windows,
)
from fit_stage_methods import apply_fitstage_to_balmer_windows

# --- Configuration ---
INPUT_DIR = Path("/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_cache")
HOTSTARS_CODE = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/HotStarsBOSS/code")
SPECFANN_DIR = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/jvlibs/SpecFANN")
BUNDLE_NAME = "MW_v1.1"
LINES_FILE = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/scripts/line_lists/fastwind_full_26.txt")
OUTPUT_DIR = Path(__file__).resolve().parent

SPECS = [
    ("109950", "spec-109950-60568-63050395314096475", "82929204"),
    ("112053", "spec-112053-60360-63050396111804682", "103295240"),
]

METHODS = {
    "standard": {"fit_stage": "standard"},
    "bsub": {"fit_stage": "balmer_subtract"},
    "sb25": {"fit_stage": "sideband_linear", "sb_half_width": 25.0},
    "sb30": {"fit_stage": "sideband_linear", "sb_half_width": 30.0},
    "sb35": {"fit_stage": "sideband_linear", "sb_half_width": 35.0},
}

# Normalisation parameters
NORM_KW = dict(
    win_kms=1200.0,
    min_win_px=21,
    sigma_lower=2.0,
    sigma_upper=3.0,
    n_iter=3,
    spline_order=3,
    smooth_scale=50.0,
    use_mad=True,
    telluric_masks=None,
    eps=1e-12,
)


def load_lines(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in raw if l.strip() and not l.strip().startswith("#")]


def process_one(
    spec_name: str,
    field: str,
    method_name: str,
    method_cfg: dict,
    *,
    groups,
    line_windows,
    continuum_iterative_sigma_clip,
) -> list[dict]:
    fits_path = INPUT_DIR / field / f"{spec_name}.fits"
    wave, flux, ivar = load_spectrum(fits_path)
    base_mask = np.isfinite(wave) & np.isfinite(flux)
    if ivar is not None:
        base_mask &= np.isfinite(ivar) & (ivar > 0)

    balmer_mask = {"k": 1.5, "stage": "fit", "smooth_sigma_px": 2.0}

    window_results, _ = local_normalise_windows(
        wave, flux, ivar,
        groups=groups,
        line_windows=line_windows,
        continuum_iterative_sigma_clip=continuum_iterative_sigma_clip,
        base_mask=base_mask,
        min_points=21,
        balmer_mask=balmer_mask,
        **NORM_KW,
    )

    fit_stage = method_cfg["fit_stage"]
    if fit_stage != "standard":
        fs_kwargs: dict = {}
        if fit_stage in ("balmer_subtract", "sideband_linear"):
            fs_kwargs["continuum_fit_func"] = continuum_iterative_sigma_clip
            fs_kwargs["continuum_fit_kwargs"] = NORM_KW.copy()
        if fit_stage == "balmer_subtract":
            fs_kwargs["use_fwhm_mask"] = True
        if fit_stage == "sideband_linear":
            fs_kwargs["sb_half_width"] = method_cfg.get("sb_half_width", 30.0)
        apply_fitstage_to_balmer_windows(
            fit_stage, window_results, line_windows, **fs_kwargs,
        )

    # Apply p98 renorm
    for result in window_results:
        wave_win = result["wave"]
        flux_win = result["flux"]
        cont = result["continuum"]
        base_mask_win = result["base_mask"]
        extra_mask = result.get("fwhm_mask")
        cont = renorm_p98_continuum(
            wave_win, flux_win, cont,
            base_mask=base_mask_win,
            line_range=(result["lo"], result["hi"]),
            renorm_percentile=98.0,
            spike_sigma=4.0,
            smooth_sigma_px=2.0,
            extra_mask=extra_mask,
        )
        result["continuum"] = cont
        result["flux_norm"] = flux_win / np.clip(cont, 1e-12, None)

    return window_results


def find_hdelta_window(window_results: list[dict]) -> dict | None:
    for r in window_results:
        lines = [str(l) for l in r.get("lines", [])]
        if "HDELTA" in lines:
            return r
    return None


def plot_comparison(sdss_id: str, results_by_method: dict[str, dict], out_path: Path):
    colours = {
        "standard": ("green", "-"),
        "bsub": ("red", "-"),
        "sb25": ("blue", "--"),
        "sb30": ("blue", "-"),
        "sb35": ("blue", ":"),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [1.2, 1]})

    # Raw flux (same for all methods — plot once)
    first = next(iter(results_by_method.values()))
    wave = np.asarray(first["wave"])
    flux = np.asarray(first["flux"])
    ax1.plot(wave, flux, color="gray", lw=0.5, alpha=0.7, label="raw flux")

    for method_name, r in results_by_method.items():
        w = np.asarray(r["wave"])
        cont = np.asarray(r["continuum"])
        fnorm = np.asarray(r["flux_norm"])
        c, ls = colours.get(method_name, ("black", "-"))
        ax1.plot(w, cont, color=c, ls=ls, lw=1.5, label=f"{method_name} cont")
        ax2.plot(w, fnorm, color=c, ls=ls, lw=1.0, label=f"{method_name} norm")

    ax1.set_ylabel("Flux")
    ax1.set_title(f"{sdss_id} HDELTA: standard vs bsub vs sideband-linear")
    ax1.legend(fontsize=8, ncol=2)

    ax2.axhline(1.0, color="black", ls=":", lw=0.5)
    ax2.set_ylabel("Normalised flux")
    ax2.set_xlabel("Wavelength [Å]")
    ax2.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    lines = load_lines(LINES_FILE)
    continuum_iterative_sigma_clip, _ = import_hotstars_normalisation(HOTSTARS_CODE)
    bundle_path = resolve_bundle_path(SPECFANN_DIR, BUNDLE_NAME, None)

    line_windows, _ = build_line_windows(bundle_path, lines, window_pad=2.0, rv_pad_kms=500.0)
    line_windows = restrict_hdelta_group_windows(
        bundle_path, line_windows, window_pad=2.0, rv_pad_kms=500.0,
    )
    groups = build_groups(line_windows, merge_pad=0.0, merge_windows=True)

    # Apply Balmer blue pads
    for group in groups:
        group_lines = [str(l) for l in group["lines"]]
        if not any(l in BALMER_LINES for l in group_lines):
            continue
        pad = 10.0 if "HDELTA" in group_lines else 20.0
        group["lo"] = max(0.0, float(group["lo"]) - pad)

    for field, spec_name, sdss_id in SPECS:
        print(f"Processing {sdss_id} ({spec_name})...")
        hdelta_by_method: dict[str, dict] = {}

        for method_name, method_cfg in METHODS.items():
            window_results = process_one(
                spec_name, field, method_name, method_cfg,
                groups=groups,
                line_windows=line_windows,
                continuum_iterative_sigma_clip=continuum_iterative_sigma_clip,
            )
            hd = find_hdelta_window(window_results)
            if hd is not None:
                hdelta_by_method[method_name] = hd
            else:
                print(f"  WARNING: no HDELTA window for {method_name}")

        if hdelta_by_method:
            out_path = OUTPUT_DIR / f"{sdss_id}_hdelta_sideband_comparison.png"
            plot_comparison(sdss_id, hdelta_by_method, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
