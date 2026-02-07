#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from local_normalise_boss_spectra import (
    DEFAULT_LINES,
    BALMER_CENTRES,
    build_groups,
    build_line_windows,
    estimate_fwhm_from_norm,
    import_hotstars_normalisation,
    load_spectrum,
    local_normalise_windows,
    renorm_p98_continuum,
    resolve_bundle_path,
    restrict_hdelta_group_windows,
)

DEFAULT_LOCAL_DIR = Path(
    "/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_local_norm"
)
DEFAULT_FULL_DIR = Path(
    "/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_normalised"
)
DEFAULT_INPUT_DIR = Path(
    "/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_cache"
)
DEFAULT_SPECFANN_DIR = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/jvlibs/SpecFANN")
DEFAULT_BUNDLE_NAME = "MW_v1.1"

BALMER_LINES = ["HDELTA", "HGAMMA", "HBETA", "HALPHA"]


def draw_balmer_mask(ax: plt.Axes, fwhm_meta: dict[str, object] | None, *, colour: str = "tab:purple") -> None:
    if not fwhm_meta:
        return
    lines = fwhm_meta.get("lines")
    if not isinstance(lines, dict):
        return
    for line_meta in lines.values():
        if not isinstance(line_meta, dict):
            continue
        mask_lo = line_meta.get("mask_lo")
        mask_hi = line_meta.get("mask_hi")
        if mask_lo is None or mask_hi is None:
            continue
        ax.axvspan(float(mask_lo), float(mask_hi), color=colour, alpha=0.15, lw=0.0)


def lorentzian_continuum_model(
    wave: np.ndarray,
    cont0: float,
    cont1: float,
    depth: float,
    centre: float,
    gamma: float,
) -> np.ndarray:
    return cont0 + cont1 * (wave - centre) - depth / (1.0 + ((wave - centre) / gamma) ** 2)


def fit_lorentzian_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    fit_range: tuple[float, float],
    ivar: np.ndarray | None = None,
    fwhm_hint: float | None = None,
    centre_hint: float | None = None,
) -> dict[str, object] | None:
    lo, hi = fit_range
    mask = (wave >= lo) & (wave <= hi) & np.isfinite(wave) & np.isfinite(flux)
    if ivar is not None:
        mask &= np.isfinite(ivar) & (ivar > 0)
    if mask.sum() < 5:
        return None
    w = wave[mask]
    f = flux[mask]
    if centre_hint is None:
        centre_hint = float(w[int(np.argmin(f))])

    cont0 = float(np.nanmedian(f))
    cont1 = 0.0
    depth = float(max(cont0 - float(np.nanmin(f)), 1e-3))
    if fwhm_hint is None or not np.isfinite(fwhm_hint):
        fwhm_hint = max(0.5, 0.2 * (hi - lo))
    gamma = max(0.2, float(fwhm_hint) / 2.0)

    p0 = [cont0, cont1, depth, float(centre_hint), gamma]
    lower = [0.0, -np.inf, 0.0, float(lo), 1e-3]
    upper = [np.inf, np.inf, cont0 * 2.0, float(hi), (hi - lo)]

    sigma = None
    if ivar is not None:
        sigma = np.where(ivar[mask] > 0, 1.0 / np.sqrt(ivar[mask]), np.inf)
    try:
        params, _ = curve_fit(
            lorentzian_continuum_model,
            w,
            f,
            p0=p0,
            bounds=(lower, upper),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=10_000,
        )
    except Exception:
        return None

    model = lorentzian_continuum_model(w, *params)
    cont_line = params[0] + params[1] * (w - params[3])
    return {
        "params": params,
        "wave": w,
        "model": model,
        "continuum": cont_line,
        "fwhm": float(2.0 * params[4]),
    }


def load_registry_first(local_dir: Path) -> tuple[str, str]:
    registry = local_dir / "processed_registry.parquet"
    if not registry.exists():
        raise FileNotFoundError(f"Missing registry: {registry}")
    import pandas as pd

    df = pd.read_parquet(registry)
    if df.empty:
        raise RuntimeError("Registry is empty; no spectra available.")
    row = df.iloc[0]
    field = str(row["field"])
    spec_file = str(row["spec_file"])
    return field, spec_file


def normalise_field_name(field: str) -> str:
    return field.zfill(6) if field.isdigit() else field


def read_full_spectrum(full_file: Path, spec_name: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(full_file, "r") as h5f:
        if spec_name not in h5f:
            raise KeyError(f"{spec_name} not found in {full_file}")
        grp = h5f[spec_name]
        wave = np.asarray(grp["wave"], dtype=float)
        flux = np.asarray(grp["flux"], dtype=float)
    return wave, flux


def resample_full_to_window(wave_full: np.ndarray, flux_full: np.ndarray, wave_win: np.ndarray) -> np.ndarray:
    mask = np.isfinite(wave_full) & np.isfinite(flux_full)
    if np.count_nonzero(mask) < 2:
        return np.full_like(wave_win, np.nan, dtype=float)
    return np.interp(wave_win, wave_full[mask], flux_full[mask], left=np.nan, right=np.nan)


def compute_local_windows(
    *,
    fits_path: Path,
    specfann_dir: Path,
    bundle_name: str,
    lines: list[str],
    balmer_blue_pad: float,
    hdelta_blue_pad: float | None,
    window_pad: float,
    rv_pad_kms: float,
    merge_windows: bool,
    balmer_mask_k: float | None,
    balmer_mask_stage: str,
    balmer_mask_smooth: float,
    win_kms: float,
    min_win_px: int,
    sigma_lower: float,
    sigma_upper: float,
    n_iter: int,
    spline_order: int,
    smooth_scale: float,
    use_mad: bool,
    p98_renorm: bool,
    renorm_percentile: float,
    spike_sigma: float,
    smooth_sigma_px: float,
) -> dict[str, dict[str, object]]:
    continuum_iterative_sigma_clip, _ = import_hotstars_normalisation(
        Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/HotStarsBOSS/code")
    )

    bundle_path = resolve_bundle_path(specfann_dir, bundle_name, None)
    windows, _ = build_line_windows(bundle_path, lines, window_pad=window_pad, rv_pad_kms=rv_pad_kms)
    windows = restrict_hdelta_group_windows(bundle_path, windows, window_pad=window_pad, rv_pad_kms=rv_pad_kms)
    groups = build_groups(windows, merge_pad=0.0, merge_windows=merge_windows)
    if balmer_blue_pad > 0 or (hdelta_blue_pad is not None and hdelta_blue_pad > 0):
        for group in groups:
            group_lines = [str(item) for item in group["lines"]]
            if not any(line in BALMER_LINES for line in group_lines):
                continue
            pad = balmer_blue_pad
            if "HDELTA" in group_lines and hdelta_blue_pad is not None:
                pad = hdelta_blue_pad
            if pad > 0:
                group["lo"] = max(0.0, float(group["lo"]) - pad)

    wave, flux, ivar = load_spectrum(fits_path)
    base_mask = np.isfinite(wave) & np.isfinite(flux)
    if ivar is not None:
        base_mask &= np.isfinite(ivar) & (ivar > 0)

    balmer_mask = None
    if balmer_mask_k is not None and balmer_mask_k > 0:
        balmer_mask = {
            "k": float(balmer_mask_k),
            "stage": str(balmer_mask_stage),
            "smooth_sigma_px": float(balmer_mask_smooth),
        }

    window_results, _ = local_normalise_windows(
        wave,
        flux,
        ivar,
        groups=groups,
        line_windows=windows,
        continuum_iterative_sigma_clip=continuum_iterative_sigma_clip,
        base_mask=base_mask,
        min_points=min_win_px,
        win_kms=win_kms,
        min_win_px=min_win_px,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        n_iter=n_iter,
        spline_order=spline_order,
        smooth_scale=smooth_scale,
        use_mad=use_mad,
        telluric_masks=None,
        eps=1e-12,
        balmer_mask=balmer_mask,
    )

    if p98_renorm:
        for result in window_results:
            extra_mask = None
            if balmer_mask is not None and balmer_mask_stage in {"p98", "fit+p98"}:
                extra_mask = result.get("fwhm_mask")
            cont = renorm_p98_continuum(
                result["wave"],
                result["flux"],
                result["continuum"],
                base_mask=result["base_mask"],
                line_range=(result["lo"], result["hi"]),
                renorm_percentile=renorm_percentile,
                spike_sigma=spike_sigma,
                smooth_sigma_px=smooth_sigma_px,
                extra_mask=extra_mask,
            )
            result["continuum"] = cont
            result["flux_norm"] = result["flux"] / np.clip(cont, 1e-12, None)

    by_line: dict[str, dict[str, object]] = {}
    for result in window_results:
        result_lines = [str(item) for item in result["lines"]]
        for line in BALMER_LINES:
            if line in result_lines and line not in by_line:
                line_ranges = result.get("line_ranges")
                if not isinstance(line_ranges, dict):
                    line_ranges = {}
                    result["line_ranges"] = line_ranges
                if line in windows:
                    line_ranges[line] = windows[line]
                by_line[line] = result
    return by_line


def plot_balmer_comparison(
    *,
    spec_name: str,
    windows: dict[str, dict[str, object]],
    wave_full: np.ndarray,
    flux_full: np.ndarray,
    output_path: Path,
) -> None:
    n = len(BALMER_LINES)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 2.6 * n), sharex=False)
    for idx, line in enumerate(BALMER_LINES):
        ax_raw, ax_norm = axes[idx]
        if line not in windows:
            ax_raw.text(0.5, 0.5, f"{line} missing", ha="center", va="center")
            ax_norm.axis("off")
            continue
        result = windows[line]
        wave = result["wave"]
        flux = result["flux"]
        cont = result["continuum"]
        flux_norm = result["flux_norm"]
        flux_full_win = resample_full_to_window(wave_full, flux_full, wave)
        cont_full = np.full_like(flux, np.nan, dtype=float)
        m_cont = np.isfinite(flux) & np.isfinite(flux_full_win) & (flux_full_win != 0)
        cont_full[m_cont] = flux[m_cont] / flux_full_win[m_cont]
        fwhm_meta = result.get("fwhm_meta")

        draw_balmer_mask(ax_raw, fwhm_meta)
        ax_raw.plot(wave, flux, color="k", lw=1.0, label="Raw flux")
        ax_raw.plot(wave, cont, color="tab:orange", lw=1.4, label="Local continuum")
        ax_raw.plot(wave, cont_full, color="tab:blue", lw=1.2, label="Full-spectrum continuum")
        ax_raw.set_ylabel("Flux")
        ax_raw.set_title(f"{line}: raw + continuum")
        ax_raw.legend(loc="best", fontsize=8)

        draw_balmer_mask(ax_norm, fwhm_meta)
        ax_norm.plot(wave, flux_full_win, color="black", lw=1.0, label="Full norm")
        ax_norm.plot(wave, flux_norm, color="tab:red", lw=1.0, label="Local norm")
        ax_norm.axhline(1.0, ls="--", color="grey", alpha=0.6, lw=0.8)
        ax_norm.set_ylabel("Normalised flux")
        ax_norm.set_title(f"{line}: local vs full")
        ax_norm.legend(loc="best", fontsize=8)
        ax_norm.set_ylim(0.4, 1.3)

        if idx == n - 1:
            ax_raw.set_xlabel("Wavelength (A)")
            ax_norm.set_xlabel("Wavelength (A)")

    fig.suptitle(f"{spec_name} — Balmer window diagnostics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lorentzian_fits(
    *,
    spec_name: str,
    windows: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    n = len(BALMER_LINES)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 2.3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for idx, line in enumerate(BALMER_LINES):
        ax = axes[idx]
        if line not in windows:
            ax.text(0.5, 0.5, f"{line} missing", ha="center", va="center")
            ax.axis("off")
            continue

        result = windows[line]
        wave = result["wave"]
        flux = result["flux"]
        ivar = result.get("ivar")
        cont = result["continuum"]
        flux_norm = result["flux_norm"]
        fwhm_meta = result.get("fwhm_meta")
        line_ranges = result.get("line_ranges", {})
        fit_range = line_ranges.get(line, (result["lo"], result["hi"]))

        fwhm_hint, centre_hint, _ = estimate_fwhm_from_norm(
            wave,
            flux_norm,
            fit_range,
            centre_hint=BALMER_CENTRES.get(line),
            smooth_sigma_px=2.0,
        )
        fit = fit_lorentzian_continuum(
            wave,
            flux,
            fit_range=fit_range,
            ivar=ivar,
            fwhm_hint=fwhm_hint,
            centre_hint=centre_hint,
        )

        draw_balmer_mask(ax, fwhm_meta)
        ax.plot(wave, flux, color="k", lw=0.9, label="Raw flux")
        ax.plot(wave, cont, color="tab:orange", lw=1.2, label="Local continuum")
        if fit is not None:
            ax.plot(fit["wave"], fit["model"], color="tab:blue", lw=1.1, label="Lorentzian+cont")
            ax.plot(fit["wave"], fit["continuum"], color="tab:green", lw=1.0, ls="--", label="Fit continuum")
            ax.set_title(f"{line}: FWHM~{fit['fwhm']:.1f} A")
        else:
            ax.set_title(f"{line}: fit failed")
        ax.set_ylabel("Flux")
        if idx == 0:
            ax.legend(loc="best", fontsize=8)
        if idx == n - 1:
            ax.set_xlabel("Wavelength (A)")

    fig.suptitle(f"{spec_name} — Lorentzian fit diagnostics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Balmer diagnostics: local vs full normalisation.")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--full-dir", type=Path, default=DEFAULT_FULL_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--specfann-dir", type=Path, default=DEFAULT_SPECFANN_DIR)
    parser.add_argument("--bundle-name", type=str, default=DEFAULT_BUNDLE_NAME)
    parser.add_argument(
        "--lines-file",
        type=Path,
        default=None,
        help="Line list file (default: local normalisation default list).",
    )
    parser.add_argument("--field", type=str, default=None)
    parser.add_argument("--spec-file", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--balmer-blue-pad",
        type=float,
        default=0.0,
        help="Extra padding (A) on the blue wing of Balmer windows.",
    )
    parser.add_argument(
        "--hdelta-blue-pad",
        type=float,
        default=None,
        help="Override blue-wing padding (A) for HDELTA only.",
    )
    parser.add_argument(
        "--balmer-mask-k",
        type=float,
        default=None,
        help="Mask +/- k*FWHM around Balmer cores when computing the local continuum.",
    )
    parser.add_argument(
        "--balmer-mask-stage",
        type=str,
        default="fit",
        choices=["fit", "p98", "fit+p98"],
        help="Apply the Balmer FWHM mask during the spline fit, p98 renorm, or both.",
    )
    parser.add_argument(
        "--balmer-mask-smooth",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma (pixels) for Balmer FWHM estimation.",
    )
    parser.add_argument("--window-pad", type=float, default=2.0)
    parser.add_argument("--rv-pad-kms", type=float, default=500.0)
    parser.add_argument(
        "--merge-windows",
        action="store_true",
        default=True,
        help="Merge overlapping windows before fitting (default: on).",
    )
    parser.add_argument(
        "--no-merge-windows",
        dest="merge_windows",
        action="store_false",
        help="Do not merge overlapping windows.",
    )
    parser.add_argument("--win-kms", type=float, default=1200.0)
    parser.add_argument("--min-win-px", type=int, default=21)
    parser.add_argument("--sigma-lower", type=float, default=2.0)
    parser.add_argument("--sigma-upper", type=float, default=3.0)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--smooth-scale", type=float, default=50.0)
    parser.add_argument("--use-mad", action="store_true", default=True)
    parser.add_argument("--no-use-mad", dest="use_mad", action="store_false")
    parser.add_argument("--p98-renorm", action="store_true", default=True)
    parser.add_argument("--no-p98-renorm", dest="p98_renorm", action="store_false")
    parser.add_argument("--renorm-percentile", type=float, default=98.0)
    parser.add_argument("--spike-sigma", type=float, default=4.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument(
        "--lorentzian-plots",
        action="store_true",
        help="Write Lorentzian fit diagnostic plots alongside the Balmer comparison.",
    )
    args = parser.parse_args()

    if args.field is None or args.spec_file is None:
        field, spec_file = load_registry_first(args.local_dir)
    else:
        field = args.field
        spec_file = args.spec_file

    field = normalise_field_name(field)
    spec_name = Path(spec_file).stem

    local_file = args.local_dir / f"{field}.h5"
    full_file = args.full_dir / f"{field}.h5"
    fits_path = args.input_dir / field / spec_file
    if not local_file.exists():
        raise FileNotFoundError(f"Local file not found: {local_file}")
    if not full_file.exists():
        raise FileNotFoundError(f"Full file not found: {full_file}")
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS not found: {fits_path}")

    wave_full, flux_full = read_full_spectrum(full_file, spec_name)
    if args.lines_file is not None:
        raw = args.lines_file.read_text(encoding="utf-8").splitlines()
        lines = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    else:
        lines = DEFAULT_LINES.copy()

    windows = compute_local_windows(
        fits_path=fits_path,
        specfann_dir=args.specfann_dir,
        bundle_name=args.bundle_name,
        lines=lines,
        balmer_blue_pad=args.balmer_blue_pad,
        hdelta_blue_pad=args.hdelta_blue_pad,
        window_pad=args.window_pad,
        rv_pad_kms=args.rv_pad_kms,
        merge_windows=args.merge_windows,
        balmer_mask_k=args.balmer_mask_k,
        balmer_mask_stage=args.balmer_mask_stage,
        balmer_mask_smooth=args.balmer_mask_smooth,
        win_kms=args.win_kms,
        min_win_px=args.min_win_px,
        sigma_lower=args.sigma_lower,
        sigma_upper=args.sigma_upper,
        n_iter=args.n_iter,
        spline_order=args.spline_order,
        smooth_scale=args.smooth_scale,
        use_mad=args.use_mad,
        p98_renorm=args.p98_renorm,
        renorm_percentile=args.renorm_percentile,
        spike_sigma=args.spike_sigma,
        smooth_sigma_px=args.smooth_sigma,
    )

    out_path = args.output
    if out_path is None:
        out_dir = args.local_dir / "figures"
        suffix = ""
        if args.balmer_blue_pad > 0 or (args.hdelta_blue_pad is not None and args.hdelta_blue_pad > 0):
            pad_label = f"{args.balmer_blue_pad:g}".replace(".", "p")
            suffix = f"_bluepad{pad_label}"
            if args.hdelta_blue_pad is not None and args.hdelta_blue_pad != args.balmer_blue_pad:
                hdelta_label = f"{args.hdelta_blue_pad:g}".replace(".", "p")
                suffix = f"{suffix}_hdelta{hdelta_label}"
        if args.balmer_mask_k is not None and args.balmer_mask_k > 0:
            mask_label = f"k{args.balmer_mask_k:g}".replace(".", "p")
            stage_label = args.balmer_mask_stage.replace("+", "p")
            suffix = f"{suffix}_fwhm{mask_label}_{stage_label}"
        if suffix:
            out_dir = out_dir / suffix.lstrip("_")
        out_path = out_dir / f"{spec_name}_balmer_local_vs_full{suffix}.png"

    plot_balmer_comparison(
        spec_name=spec_name,
        windows=windows,
        wave_full=wave_full,
        flux_full=flux_full,
        output_path=out_path,
    )
    if args.lorentzian_plots:
        lorentz_path = out_path.with_name(
            out_path.name.replace("balmer_local_vs_full", "balmer_lorentzian_fits")
        )
        plot_lorentzian_fits(
            spec_name=spec_name,
            windows=windows,
            output_path=lorentz_path,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
