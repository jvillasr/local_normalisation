#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from local_normalise_boss_spectra import (  # type: ignore
    BALMER_CENTRES,
    BALMER_LINES,
    DEFAULT_LINES,
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
from plot_balmer_local_vs_full import (  # type: ignore
    fit_lorentzian_continuum,
    plot_balmer_comparison,
    plot_lorentzian_fits,
)

DEFAULT_INPUT_DIR = Path(
    "/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_cache"
)
DEFAULT_FULL_DIR = Path(
    "/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_normalised"
)
DEFAULT_PILOT_SOURCE_DIR = Path(
    "/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_local_norm/figures/bluepad20_hdelta10_fwhmk1_fit"
)
DEFAULT_SPECFANN_DIR = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/jvlibs/SpecFANN")
DEFAULT_BUNDLE_NAME = "MW_v1.1"
DEFAULT_LINES_FILE = Path(
    "/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/scripts/line_lists/fastwind_full_26.txt"
)
DEFAULT_HOTSTARS_CODE = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/HotStarsBOSS/code")


@dataclass(frozen=True)
class MethodConfig:
    name: str
    renorm_mode: str  # none | p98 | wing_anchor | upper_envelope
    balmer_mask_k: float
    balmer_mask_stage: str


def parse_field_from_spec_name(spec_name: str) -> str:
    parts = spec_name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Unexpected spec name: {spec_name}")
    field = parts[1]
    return field.zfill(6) if field.isdigit() else field


def list_pilot_specs(fig_dir: Path) -> list[str]:
    specs = sorted(
        {
            path.name.split("_balmer_local_vs_full_")[0]
            for path in fig_dir.glob("*_balmer_local_vs_full_*.png")
        }
    )
    if not specs:
        raise RuntimeError(f"No pilot spectra found in {fig_dir}")
    return specs


def load_lines(lines_file: Path | None) -> list[str]:
    if lines_file is None:
        return DEFAULT_LINES.copy()
    raw = lines_file.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise RuntimeError(f"No lines found in {lines_file}")
    return lines


def read_full_spectrum(full_file: Path, spec_name: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(full_file, "r") as h5f:
        if spec_name not in h5f:
            raise KeyError(f"{spec_name} not found in {full_file}")
        grp = h5f[spec_name]
        wave = np.asarray(grp["wave"], dtype=float)
        flux = np.asarray(grp["flux"], dtype=float)
    return wave, flux


def resample_full_to_window(wave_full: np.ndarray, flux_full: np.ndarray, wave_win: np.ndarray) -> np.ndarray:
    valid = np.isfinite(wave_full) & np.isfinite(flux_full)
    if np.count_nonzero(valid) < 2:
        return np.full_like(wave_win, np.nan, dtype=float)
    return np.interp(wave_win, wave_full[valid], flux_full[valid], left=np.nan, right=np.nan)


def _robust_upper_anchor(
    values: np.ndarray,
    *,
    percentile: float,
    spike_sigma: float,
) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(vals))
    if np.isfinite(sigma) and sigma > 0:
        vals = vals[vals <= med + spike_sigma * sigma]
    if vals.size == 0:
        return float("nan")
    return float(np.nanpercentile(vals, percentile))


def renorm_wing_anchor_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    *,
    base_mask: np.ndarray,
    line_range: tuple[float, float],
    centre: float,
    fwhm: float,
    extra_mask: np.ndarray | None,
    smooth_sigma_px: float,
    anchor_percentile: float,
    spike_sigma: float,
    inner_scale: float,
    outer_scale: float,
    min_points_side: int,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    meta: dict[str, float | int | str] = {
        "status": "ok",
        "centre": float(centre),
        "fwhm": float(fwhm),
    }
    if not np.isfinite(fwhm) or fwhm <= 0:
        meta["status"] = "invalid_fwhm"
        return continuum, meta

    lo, hi = line_range
    mask = base_mask & (wave >= lo) & (wave <= hi)
    if extra_mask is not None and extra_mask.shape == mask.shape:
        mask &= extra_mask

    ratio = np.full_like(flux, np.nan, dtype=float)
    valid = mask & np.isfinite(flux) & np.isfinite(continuum) & (continuum > 0)
    ratio[valid] = flux[valid] / continuum[valid]
    if smooth_sigma_px > 0:
        finite_ratio = np.isfinite(ratio)
        if np.count_nonzero(finite_ratio) > 3:
            fill = float(np.nanmedian(ratio[finite_ratio]))
            work = ratio.copy()
            work[~finite_ratio] = fill
            ratio = gaussian_filter1d(work, sigma=smooth_sigma_px, mode="nearest")
            ratio[~mask] = np.nan

    dist = np.abs(wave - centre)
    left = mask & (wave < centre) & (dist >= inner_scale * fwhm) & (dist <= outer_scale * fwhm)
    right = mask & (wave > centre) & (dist >= inner_scale * fwhm) & (dist <= outer_scale * fwhm)
    n_left = int(np.count_nonzero(left))
    n_right = int(np.count_nonzero(right))
    meta["n_left"] = n_left
    meta["n_right"] = n_right

    if n_left < min_points_side or n_right < min_points_side:
        meta["status"] = "insufficient_wings"
        return continuum, meta

    q_left = _robust_upper_anchor(ratio[left], percentile=anchor_percentile, spike_sigma=spike_sigma)
    q_right = _robust_upper_anchor(ratio[right], percentile=anchor_percentile, spike_sigma=spike_sigma)
    if not np.isfinite(q_left) or not np.isfinite(q_right):
        meta["status"] = "invalid_anchor"
        return continuum, meta

    x_left = float(np.nanmedian(wave[left] - centre))
    x_right = float(np.nanmedian(wave[right] - centre))
    if not np.isfinite(x_left) or not np.isfinite(x_right) or np.isclose(x_left, x_right):
        meta["status"] = "invalid_geometry"
        return continuum, meta

    slope = (q_right - q_left) / (x_right - x_left)
    intercept = 0.5 * ((q_left - slope * x_left) + (q_right - slope * x_right))

    corr = intercept + slope * (wave - centre)
    corr = np.clip(corr, clip_min, clip_max)

    out = continuum.copy()
    apply_mask = mask & np.isfinite(corr)
    out[apply_mask] = continuum[apply_mask] * corr[apply_mask]

    meta["anchor_left"] = float(q_left)
    meta["anchor_right"] = float(q_right)
    meta["intercept"] = float(intercept)
    meta["slope"] = float(slope)
    return out, meta


def renorm_upper_envelope_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    *,
    base_mask: np.ndarray,
    line_range: tuple[float, float],
    centre: float,
    fwhm: float,
    extra_mask: np.ndarray | None,
    smooth_sigma_px: float,
    envelope_percentile: float,
    spike_sigma: float,
    inner_scale: float,
    outer_scale: float,
    min_points: int,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    meta: dict[str, float | int | str] = {
        "status": "ok",
        "centre": float(centre),
        "fwhm": float(fwhm),
    }
    if not np.isfinite(fwhm) or fwhm <= 0:
        meta["status"] = "invalid_fwhm"
        return continuum, meta

    lo, hi = line_range
    mask = base_mask & (wave >= lo) & (wave <= hi)
    if extra_mask is not None and extra_mask.shape == mask.shape:
        mask &= extra_mask

    ratio = np.full_like(flux, np.nan, dtype=float)
    valid = mask & np.isfinite(flux) & np.isfinite(continuum) & (continuum > 0)
    ratio[valid] = flux[valid] / continuum[valid]
    if smooth_sigma_px > 0:
        finite_ratio = np.isfinite(ratio)
        if np.count_nonzero(finite_ratio) > 3:
            fill = float(np.nanmedian(ratio[finite_ratio]))
            work = ratio.copy()
            work[~finite_ratio] = fill
            ratio = gaussian_filter1d(work, sigma=smooth_sigma_px, mode="nearest")
            ratio[~mask] = np.nan

    dist = np.abs(wave - centre)
    candidate = mask & (dist >= inner_scale * fwhm) & (dist <= outer_scale * fwhm) & np.isfinite(ratio)
    n_candidate = int(np.count_nonzero(candidate))
    meta["n_candidate"] = n_candidate
    if n_candidate < max(min_points, 4):
        meta["status"] = "insufficient_wings"
        return continuum, meta

    vals = ratio[candidate]
    thresh = _robust_upper_anchor(vals, percentile=envelope_percentile, spike_sigma=spike_sigma)
    if not np.isfinite(thresh):
        meta["status"] = "invalid_anchor"
        return continuum, meta

    env = candidate & (ratio >= thresh)
    n_env = int(np.count_nonzero(env))
    if n_env < max(min_points, 4):
        cand_idx = np.where(candidate)[0]
        order = np.argsort(ratio[cand_idx])
        keep_idx = cand_idx[order[-max(min_points, 4) :]]
        env = np.zeros_like(candidate, dtype=bool)
        env[keep_idx] = True
        n_env = int(np.count_nonzero(env))
    meta["n_envelope"] = n_env
    meta["threshold"] = float(thresh)

    x = wave[env] - centre
    y = ratio[env]
    if x.size < 4:
        meta["status"] = "insufficient_envelope"
        return continuum, meta

    # Slightly favour the highest-envelope points.
    w = np.clip(y - np.nanmedian(y) + 1.0, 0.1, 10.0)
    try:
        slope, intercept = np.polyfit(x, y, deg=1, w=w)
    except Exception:
        slope = 0.0
        intercept = float(np.nanmedian(y))

    corr = intercept + slope * (wave - centre)
    corr = np.clip(corr, clip_min, clip_max)

    out = continuum.copy()
    apply_mask = mask & np.isfinite(corr)
    out[apply_mask] = continuum[apply_mask] * corr[apply_mask]

    meta["intercept"] = float(intercept)
    meta["slope"] = float(slope)
    return out, meta


def apply_method_renorm(
    method: MethodConfig,
    window_results: list[dict[str, object]],
    line_windows: dict[str, tuple[float, float]],
    *,
    renorm_percentile: float,
    spike_sigma: float,
    smooth_sigma_px: float,
    wing_inner_scale: float,
    wing_outer_scale: float,
    wing_anchor_percentile: float,
    wing_min_points_side: int,
    wing_clip_min: float,
    wing_clip_max: float,
    upper_inner_scale: float,
    upper_outer_scale: float,
    upper_envelope_percentile: float,
    upper_min_points: int,
    upper_clip_min: float,
    upper_clip_max: float,
) -> None:
    if method.renorm_mode == "none":
        return

    for result in window_results:
        wave = np.asarray(result["wave"], dtype=float)
        flux = np.asarray(result["flux"], dtype=float)
        cont = np.asarray(result["continuum"], dtype=float)
        base_mask = np.asarray(result["base_mask"], dtype=bool)
        if wave.size == 0:
            continue

        if method.renorm_mode == "p98":
            extra_mask = None
            if method.balmer_mask_stage in {"p98", "fit+p98"}:
                extra_mask = result.get("fwhm_mask")
            cont = renorm_p98_continuum(
                wave,
                flux,
                cont,
                base_mask=base_mask,
                line_range=(float(result["lo"]), float(result["hi"])),
                renorm_percentile=renorm_percentile,
                spike_sigma=spike_sigma,
                smooth_sigma_px=smooth_sigma_px,
                extra_mask=extra_mask,
            )
            result["continuum"] = cont
            result["flux_norm"] = flux / np.clip(cont, 1e-12, None)
            continue

        if method.renorm_mode not in {"wing_anchor", "upper_envelope"}:
            raise ValueError(f"Unsupported renorm mode: {method.renorm_mode}")

        group_lines = [str(item) for item in result.get("lines", [])]
        renorm_meta: dict[str, dict[str, float | int | str]] = {}

        for line in BALMER_LINES:
            if line not in group_lines:
                continue
            fit_range = line_windows.get(line)
            if fit_range is None:
                continue

            centre = BALMER_CENTRES.get(line)
            fwhm = None
            fwhm_meta = result.get("fwhm_meta")
            if isinstance(fwhm_meta, dict):
                lines_meta = fwhm_meta.get("lines")
                if isinstance(lines_meta, dict) and isinstance(lines_meta.get(line), dict):
                    line_meta = lines_meta[line]
                    fwhm = line_meta.get("fwhm")
                    centre = line_meta.get("centre", centre)

            if fwhm is None or not np.isfinite(float(fwhm)):
                fwhm_est, centre_est, _ = estimate_fwhm_from_norm(
                    wave,
                    np.asarray(result["flux_norm"], dtype=float),
                    fit_range,
                    centre_hint=centre,
                    smooth_sigma_px=smooth_sigma_px,
                )
                if fwhm_est is not None:
                    fwhm = fwhm_est
                if centre_est is not None:
                    centre = centre_est

            if centre is None or fwhm is None:
                renorm_meta[line] = {"status": "missing_fwhm"}
                continue

            extra_mask = result.get("fwhm_mask")
            if method.renorm_mode == "wing_anchor":
                cont, line_meta = renorm_wing_anchor_continuum(
                    wave,
                    flux,
                    cont,
                    base_mask=base_mask,
                    line_range=fit_range,
                    centre=float(centre),
                    fwhm=float(fwhm),
                    extra_mask=extra_mask if isinstance(extra_mask, np.ndarray) else None,
                    smooth_sigma_px=smooth_sigma_px,
                    anchor_percentile=wing_anchor_percentile,
                    spike_sigma=spike_sigma,
                    inner_scale=wing_inner_scale,
                    outer_scale=wing_outer_scale,
                    min_points_side=wing_min_points_side,
                    clip_min=wing_clip_min,
                    clip_max=wing_clip_max,
                )
            else:
                cont, line_meta = renorm_upper_envelope_continuum(
                    wave,
                    flux,
                    cont,
                    base_mask=base_mask,
                    line_range=fit_range,
                    centre=float(centre),
                    fwhm=float(fwhm),
                    extra_mask=extra_mask if isinstance(extra_mask, np.ndarray) else None,
                    smooth_sigma_px=smooth_sigma_px,
                    envelope_percentile=upper_envelope_percentile,
                    spike_sigma=spike_sigma,
                    inner_scale=upper_inner_scale,
                    outer_scale=upper_outer_scale,
                    min_points=upper_min_points,
                    clip_min=upper_clip_min,
                    clip_max=upper_clip_max,
                )
            renorm_meta[line] = line_meta

        result["continuum"] = cont
        result["flux_norm"] = flux / np.clip(cont, 1e-12, None)
        result["renorm_meta"] = renorm_meta


def build_balmer_lookup(
    window_results: list[dict[str, object]],
    line_windows: dict[str, tuple[float, float]],
) -> dict[str, dict[str, object]]:
    by_line: dict[str, dict[str, object]] = {}
    for result in window_results:
        group_lines = [str(item) for item in result.get("lines", [])]
        for line in BALMER_LINES:
            if line in group_lines and line not in by_line:
                line_ranges = result.get("line_ranges")
                if not isinstance(line_ranges, dict):
                    line_ranges = {}
                    result["line_ranges"] = line_ranges
                if line in line_windows:
                    line_ranges[line] = line_windows[line]
                by_line[line] = result
    return by_line


def derivative_jump(
    wave: np.ndarray,
    series: np.ndarray,
    boundary: float,
    *,
    left_width: float = 2.0,
    right_width: float = 2.0,
) -> float:
    valid = np.isfinite(wave) & np.isfinite(series)
    if np.count_nonzero(valid) < 6:
        return float("nan")
    w = wave[valid]
    y = series[valid]
    grad = np.gradient(y, w)
    left = (w >= boundary - left_width) & (w < boundary - 0.2)
    right = (w > boundary + 0.2) & (w <= boundary + right_width)
    if np.count_nonzero(left) < 2 or np.count_nonzero(right) < 2:
        return float("nan")
    return float(np.nanmedian(grad[right]) - np.nanmedian(grad[left]))


def measure_line_metrics(
    *,
    spec_name: str,
    field: str,
    method: MethodConfig,
    line: str,
    result: dict[str, object],
    wave_full: np.ndarray,
    flux_full: np.ndarray,
    eval_inner_fwhm: float,
    eval_outer_fwhm: float,
) -> dict[str, object]:
    wave = np.asarray(result["wave"], dtype=float)
    flux = np.asarray(result["flux"], dtype=float)
    cont_local = np.asarray(result["continuum"], dtype=float)
    flux_norm_local = np.asarray(result["flux_norm"], dtype=float)
    ivar = np.asarray(result["ivar"], dtype=float) if result.get("ivar") is not None else None

    fit_range = tuple(result.get("line_ranges", {}).get(line, (float(result["lo"]), float(result["hi"]))))
    flux_full_win = resample_full_to_window(wave_full, flux_full, wave)
    cont_full = np.full_like(flux, np.nan, dtype=float)
    ok_full = np.isfinite(flux) & np.isfinite(flux_full_win) & (flux_full_win != 0)
    cont_full[ok_full] = flux[ok_full] / flux_full_win[ok_full]

    fit = fit_lorentzian_continuum(
        wave,
        flux,
        fit_range=fit_range,
        ivar=ivar,
        fwhm_hint=None,
        centre_hint=BALMER_CENTRES.get(line),
    )
    if fit is None:
        return {
            "spec_name": spec_name,
            "field": field,
            "line": line,
            "method": method.name,
            "renorm_mode": method.renorm_mode,
            "mask_k": method.balmer_mask_k,
            "mask_stage": method.balmer_mask_stage,
            "status": "fit_failed",
            "n_wing": 0,
        }

    fit_wave = np.asarray(fit["wave"], dtype=float)
    fit_cont = np.asarray(fit["continuum"], dtype=float)
    cont_fit = np.interp(wave, fit_wave, fit_cont, left=np.nan, right=np.nan)
    centre = float(fit["params"][3])
    fwhm = float(fit["fwhm"])

    dist = np.abs(wave - centre)
    wing_mask = (
        np.isfinite(cont_fit)
        & np.isfinite(cont_local)
        & np.isfinite(cont_full)
        & np.isfinite(flux_norm_local)
        & np.isfinite(flux_full_win)
        & (wave >= fit_range[0])
        & (wave <= fit_range[1])
        & (dist >= eval_inner_fwhm * fwhm)
        & (dist <= eval_outer_fwhm * fwhm)
    )
    n_wing = int(np.count_nonzero(wing_mask))
    if n_wing < 6:
        return {
            "spec_name": spec_name,
            "field": field,
            "line": line,
            "method": method.name,
            "renorm_mode": method.renorm_mode,
            "mask_k": method.balmer_mask_k,
            "mask_stage": method.balmer_mask_stage,
            "status": "insufficient_wing_points",
            "n_wing": n_wing,
            "fit_fwhm": fwhm,
        }

    frac_local = (cont_local[wing_mask] - cont_fit[wing_mask]) / np.clip(cont_fit[wing_mask], 1e-12, None)
    frac_full = (cont_full[wing_mask] - cont_fit[wing_mask]) / np.clip(cont_fit[wing_mask], 1e-12, None)
    norm_resid = flux_norm_local[wing_mask] - flux_full_win[wing_mask]

    mask_meta = result.get("fwhm_meta")
    boundary_jumps_local: list[float] = []
    boundary_jumps_full: list[float] = []
    if isinstance(mask_meta, dict):
        line_meta = mask_meta.get("lines", {}).get(line)
        if isinstance(line_meta, dict):
            for key in ("mask_lo", "mask_hi"):
                if key not in line_meta:
                    continue
                boundary = float(line_meta[key])
                jump_local = derivative_jump(wave, cont_local, boundary)
                jump_full = derivative_jump(wave, cont_full, boundary)
                if np.isfinite(jump_local):
                    boundary_jumps_local.append(abs(jump_local))
                if np.isfinite(jump_full):
                    boundary_jumps_full.append(abs(jump_full))

    kink_local = float(np.nanmax(boundary_jumps_local)) if boundary_jumps_local else float("nan")
    kink_full = float(np.nanmax(boundary_jumps_full)) if boundary_jumps_full else float("nan")

    return {
        "spec_name": spec_name,
        "field": field,
        "line": line,
        "method": method.name,
        "renorm_mode": method.renorm_mode,
        "mask_k": method.balmer_mask_k,
        "mask_stage": method.balmer_mask_stage,
        "status": "ok",
        "n_wing": n_wing,
        "fit_fwhm": fwhm,
        "bias_local": float(np.nanmedian(frac_local)),
        "abs_bias_local": float(np.nanmedian(np.abs(frac_local))),
        "under_frac_local": float(np.nanmean(cont_local[wing_mask] < cont_fit[wing_mask])),
        "bias_full": float(np.nanmedian(frac_full)),
        "abs_bias_full": float(np.nanmedian(np.abs(frac_full))),
        "under_frac_full": float(np.nanmean(cont_full[wing_mask] < cont_fit[wing_mask])),
        "rmse_norm_vs_full": float(np.sqrt(np.nanmean(norm_resid**2))),
        "kink_local": kink_local,
        "kink_full": kink_full,
    }


def compute_summary_tables(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = metrics_df.loc[metrics_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    ok["delta_abs_bias_vs_full"] = ok["abs_bias_local"] - ok["abs_bias_full"]
    ok["delta_under_vs_full"] = ok["under_frac_local"] - ok["under_frac_full"]
    ok["kink_excess_vs_full"] = ok["kink_local"] - ok["kink_full"]

    fail = (
        metrics_df.assign(is_fail=metrics_df["status"] != "ok")
        .groupby("method", as_index=False)["is_fail"]
        .mean()
        .rename(columns={"is_fail": "fail_rate"})
    )

    summary = (
        ok.groupby("method", as_index=False)
        .agg(
            n_samples=("abs_bias_local", "size"),
            median_abs_bias_local=("abs_bias_local", "median"),
            median_under_frac_local=("under_frac_local", "median"),
            median_delta_abs_bias_vs_full=("delta_abs_bias_vs_full", "median"),
            median_delta_under_vs_full=("delta_under_vs_full", "median"),
            median_kink_excess_vs_full=("kink_excess_vs_full", "median"),
            median_rmse_norm_vs_full=("rmse_norm_vs_full", "median"),
        )
        .merge(fail, on="method", how="left")
    )

    summary["composite_score"] = (
        summary["median_abs_bias_local"]
        + 0.5 * summary["median_under_frac_local"]
        + 0.2 * np.clip(summary["median_kink_excess_vs_full"], 0.0, np.inf)
        + 0.2 * summary["fail_rate"].fillna(0.0)
    )
    summary = summary.sort_values("composite_score", ascending=True).reset_index(drop=True)

    line_summary = (
        ok.groupby(["method", "line"], as_index=False)
        .agg(
            median_abs_bias_local=("abs_bias_local", "median"),
            median_under_frac_local=("under_frac_local", "median"),
            median_delta_abs_bias_vs_full=("delta_abs_bias_vs_full", "median"),
            median_kink_excess_vs_full=("kink_excess_vs_full", "median"),
            n_samples=("abs_bias_local", "size"),
        )
        .sort_values(["line", "method"])
        .reset_index(drop=True)
    )
    return summary, line_summary


def plot_metric_panels(metrics_df: pd.DataFrame, output_path: Path) -> None:
    ok = metrics_df.loc[metrics_df["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["delta_abs_bias_vs_full"] = ok["abs_bias_local"] - ok["abs_bias_full"]
    ok["kink_excess_vs_full"] = ok["kink_local"] - ok["kink_full"]

    methods = list(dict.fromkeys(ok["method"].tolist()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("abs_bias_local", "Median |continuum bias| vs Lorentzian continuum"),
        ("under_frac_local", "Fraction local continuum below Lorentzian continuum"),
        ("delta_abs_bias_vs_full", "Delta |bias| vs full normalisation (negative is better)"),
        ("kink_excess_vs_full", "Kink excess vs full normalisation (negative is better)"),
    ]

    for ax, (col, title) in zip(axes.ravel(), metrics):
        data = [ok.loc[ok["method"] == method, col].dropna().values for method in methods]
        ax.boxplot(data, tick_labels=methods, showfliers=False)
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_composite_scores(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(summary_df["method"], summary_df["composite_score"], color="tab:blue", alpha=0.8)
    ax.set_ylabel("Composite score (lower is better)")
    ax.set_title("Method ranking")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Balmer renorm methods on pilot spectra.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--full-dir", type=Path, default=DEFAULT_FULL_DIR)
    parser.add_argument("--pilot-source-dir", type=Path, default=DEFAULT_PILOT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--spec-names",
        nargs="*",
        default=None,
        help="Optional explicit spectrum names (spec-...) to evaluate.",
    )
    parser.add_argument(
        "--max-specs",
        type=int,
        default=None,
        help="Optional cap on number of pilot spectra after filtering.",
    )
    parser.add_argument("--hotstars-code", type=Path, default=DEFAULT_HOTSTARS_CODE)
    parser.add_argument("--specfann-dir", type=Path, default=DEFAULT_SPECFANN_DIR)
    parser.add_argument("--bundle-name", type=str, default=DEFAULT_BUNDLE_NAME)
    parser.add_argument("--lines-file", type=Path, default=DEFAULT_LINES_FILE)

    parser.add_argument("--balmer-blue-pad", type=float, default=20.0)
    parser.add_argument("--hdelta-blue-pad", type=float, default=10.0)
    parser.add_argument("--window-pad", type=float, default=2.0)
    parser.add_argument("--rv-pad-kms", type=float, default=500.0)
    parser.add_argument("--merge-windows", action="store_true", default=True)

    parser.add_argument("--win-kms", type=float, default=1200.0)
    parser.add_argument("--min-win-px", type=int, default=21)
    parser.add_argument("--sigma-lower", type=float, default=2.0)
    parser.add_argument("--sigma-upper", type=float, default=3.0)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--smooth-scale", type=float, default=50.0)
    parser.add_argument("--use-mad", action="store_true", default=True)

    parser.add_argument("--renorm-percentile", type=float, default=98.0)
    parser.add_argument("--spike-sigma", type=float, default=4.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)

    parser.add_argument("--wing-inner-scale", type=float, default=0.8)
    parser.add_argument("--wing-outer-scale", type=float, default=2.5)
    parser.add_argument("--wing-anchor-percentile", type=float, default=95.0)
    parser.add_argument("--wing-min-points-side", type=int, default=8)
    parser.add_argument("--wing-clip-min", type=float, default=0.75)
    parser.add_argument("--wing-clip-max", type=float, default=1.35)
    parser.add_argument("--upper-inner-scale", type=float, default=0.8)
    parser.add_argument("--upper-outer-scale", type=float, default=2.5)
    parser.add_argument("--upper-envelope-percentile", type=float, default=88.0)
    parser.add_argument("--upper-min-points", type=int, default=10)
    parser.add_argument("--upper-clip-min", type=float, default=0.85)
    parser.add_argument("--upper-clip-max", type=float, default=1.20)

    parser.add_argument("--eval-inner-fwhm", type=float, default=0.6)
    parser.add_argument("--eval-outer-fwhm", type=float, default=1.8)

    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=[
            "p98_k1p5_fit:p98:1.5:fit",
            "p98_k2_fit:p98:2.0:fit",
            "none_k1p5_fit:none:1.5:fit",
            "wing_k1p5_fit:wing_anchor:1.5:fit",
            "wing_k2_fit:wing_anchor:2.0:fit",
            "upper_k1p5_fit:upper_envelope:1.5:fit",
            "upper_k2_fit:upper_envelope:2.0:fit",
        ],
        help="Method specs as name:renorm_mode:mask_k:mask_stage",
    )
    parser.add_argument("--make-star-plots", action="store_true", default=True)
    parser.add_argument("--no-make-star-plots", dest="make_star_plots", action="store_false")
    args = parser.parse_args()

    methods: list[MethodConfig] = []
    for spec in args.methods:
        name, mode, k, stage = spec.split(":")
        methods.append(MethodConfig(name=name, renorm_mode=mode, balmer_mask_k=float(k), balmer_mask_stage=stage))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    pilot_specs = list_pilot_specs(args.pilot_source_dir)
    if args.spec_names:
        requested = [item.strip() for item in args.spec_names if item.strip()]
        requested_set = set(requested)
        pilot_specs = [spec for spec in pilot_specs if spec in requested_set]
        missing = [spec for spec in requested if spec not in set(pilot_specs)]
        if missing:
            print(f"[warn] requested spectra not found in pilot source: {', '.join(missing)}")
    if args.max_specs is not None:
        pilot_specs = pilot_specs[: max(0, int(args.max_specs))]
    if not pilot_specs:
        raise RuntimeError("No spectra selected for evaluation.")
    lines = load_lines(args.lines_file)

    continuum_iterative_sigma_clip, _ = import_hotstars_normalisation(args.hotstars_code)
    bundle_path = resolve_bundle_path(args.specfann_dir, args.bundle_name, None)
    line_windows, _ = build_line_windows(
        bundle_path,
        lines,
        window_pad=args.window_pad,
        rv_pad_kms=args.rv_pad_kms,
    )
    line_windows = restrict_hdelta_group_windows(
        bundle_path,
        line_windows,
        window_pad=args.window_pad,
        rv_pad_kms=args.rv_pad_kms,
    )
    groups = build_groups(line_windows, merge_pad=0.0, merge_windows=args.merge_windows)
    if args.balmer_blue_pad > 0 or args.hdelta_blue_pad > 0:
        for group in groups:
            group_lines = [str(item) for item in group["lines"]]
            if not any(line in BALMER_LINES for line in group_lines):
                continue
            pad = args.hdelta_blue_pad if ("HDELTA" in group_lines and args.hdelta_blue_pad is not None) else args.balmer_blue_pad
            if pad > 0:
                group["lo"] = max(0.0, float(group["lo"]) - pad)

    records: list[dict[str, object]] = []

    for spec_name in pilot_specs:
        field = parse_field_from_spec_name(spec_name)
        spec_file = f"{spec_name}.fits"
        fits_path = args.input_dir / field / spec_file
        full_file = args.full_dir / f"{field}.h5"

        if not fits_path.exists() or not full_file.exists():
            print(f"[skip] missing input/full for {spec_name}")
            continue

        try:
            wave_full, flux_full = read_full_spectrum(full_file, spec_name)
        except Exception as exc:
            print(f"[skip] failed reading full spectrum {spec_name}: {exc}")
            continue

        wave, flux, ivar = load_spectrum(fits_path)
        base_mask = np.isfinite(wave) & np.isfinite(flux)
        if ivar is not None:
            base_mask &= np.isfinite(ivar) & (ivar > 0)

        for method in methods:
            balmer_mask = {
                "k": method.balmer_mask_k,
                "stage": method.balmer_mask_stage,
                "smooth_sigma_px": args.smooth_sigma,
            }
            window_results, _ = local_normalise_windows(
                wave,
                flux,
                ivar,
                groups=groups,
                line_windows=line_windows,
                continuum_iterative_sigma_clip=continuum_iterative_sigma_clip,
                base_mask=base_mask,
                min_points=args.min_win_px,
                win_kms=args.win_kms,
                min_win_px=args.min_win_px,
                sigma_lower=args.sigma_lower,
                sigma_upper=args.sigma_upper,
                n_iter=args.n_iter,
                spline_order=args.spline_order,
                smooth_scale=args.smooth_scale,
                use_mad=args.use_mad,
                telluric_masks=None,
                eps=1e-12,
                balmer_mask=balmer_mask,
            )

            apply_method_renorm(
                method,
                window_results,
                line_windows,
                renorm_percentile=args.renorm_percentile,
                spike_sigma=args.spike_sigma,
                smooth_sigma_px=args.smooth_sigma,
                wing_inner_scale=args.wing_inner_scale,
                wing_outer_scale=args.wing_outer_scale,
                wing_anchor_percentile=args.wing_anchor_percentile,
                wing_min_points_side=args.wing_min_points_side,
                wing_clip_min=args.wing_clip_min,
                wing_clip_max=args.wing_clip_max,
                upper_inner_scale=args.upper_inner_scale,
                upper_outer_scale=args.upper_outer_scale,
                upper_envelope_percentile=args.upper_envelope_percentile,
                upper_min_points=args.upper_min_points,
                upper_clip_min=args.upper_clip_min,
                upper_clip_max=args.upper_clip_max,
            )

            windows = build_balmer_lookup(window_results, line_windows)

            if args.make_star_plots:
                method_dir = args.output_dir / "plots" / method.name
                method_dir.mkdir(parents=True, exist_ok=True)
                out_path = method_dir / f"{spec_name}_balmer_local_vs_full_{method.name}.png"
                plot_balmer_comparison(
                    spec_name=f"{spec_name} [{method.name}]",
                    windows=windows,
                    wave_full=wave_full,
                    flux_full=flux_full,
                    output_path=out_path,
                )
                lorentz_path = method_dir / f"{spec_name}_balmer_lorentzian_fits_{method.name}.png"
                plot_lorentzian_fits(
                    spec_name=f"{spec_name} [{method.name}]",
                    windows=windows,
                    output_path=lorentz_path,
                )

            for line in BALMER_LINES:
                if line not in windows:
                    records.append(
                        {
                            "spec_name": spec_name,
                            "field": field,
                            "line": line,
                            "method": method.name,
                            "renorm_mode": method.renorm_mode,
                            "mask_k": method.balmer_mask_k,
                            "mask_stage": method.balmer_mask_stage,
                            "status": "line_missing",
                            "n_wing": 0,
                        }
                    )
                    continue

                metrics = measure_line_metrics(
                    spec_name=spec_name,
                    field=field,
                    method=method,
                    line=line,
                    result=windows[line],
                    wave_full=wave_full,
                    flux_full=flux_full,
                    eval_inner_fwhm=args.eval_inner_fwhm,
                    eval_outer_fwhm=args.eval_outer_fwhm,
                )
                records.append(metrics)

        print(f"[done] {spec_name}")

    metrics_df = pd.DataFrame.from_records(records)
    metrics_path = args.output_dir / "metrics_by_line.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary_df, line_summary_df = compute_summary_tables(metrics_df)
    summary_path = args.output_dir / "summary_by_method.csv"
    line_summary_path = args.output_dir / "summary_by_method_line.csv"
    summary_df.to_csv(summary_path, index=False)
    line_summary_df.to_csv(line_summary_path, index=False)

    plot_metric_panels(metrics_df, args.output_dir / "method_metric_boxplots.png")
    plot_composite_scores(summary_df, args.output_dir / "method_composite_scores.png")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote line summary: {line_summary_path}")


if __name__ == "__main__":
    main()
