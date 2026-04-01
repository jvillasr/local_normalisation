#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_stage_methods import apply_fitstage_to_balmer_windows  # type: ignore


C_KMS = 299_792.458

DEFAULT_HOTSTARS_CODE = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/HotStarsBOSS/code")
DEFAULT_SPECFANN_DIR = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/jvlibs/SpecFANN")
DEFAULT_BUNDLE_NAME = "MW_v1.1"
DEFAULT_INPUT_DIR = Path(
    "/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_cache"
)
DEFAULT_OUTPUT_DIR = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/local_normalised_data")

DEFAULT_LINES = [
    "HDELTA",
    "HGAMMA",
    "HBETA",
    "HALPHA",
    "HEI4026",
    "HEI4143",
    "HEI4387",
    "HEI4471",
    "HEI4922",
    "HEI5016",
    "HEI5048",
    "HEI5875",
    "HEI6678",
    "HEI7065",
    "HEII4200",
    "HEII4541",
    "HEII4686",
    "HEII5411",
    "HEII6406",
    "NII3995",
    "CIII4186",
    "CIII4650",
    "CIV5801",
    "CIV5811",
    "NIII4379",
    "NIIItrip",
]
BALMER_LINES = ["HDELTA", "HGAMMA", "HBETA", "HALPHA"]
BALMER_CENTRES = {
    "HDELTA": 4101.734,
    "HGAMMA": 4340.472,
    "HBETA": 4861.333,
    "HALPHA": 6562.79,
}
HDELTA_GROUP_BASE_LINES = ["HDELTA", "HEI4143", "HEI4144"]
HGAMMA_GROUP_BASE_LINES = ["HGAMMA", "HEI4387", "NIII4379"]
HEI4471_GROUP_BASE_LINES = ["HEI4471", "HEII4541"]
HEII4686_GROUP_BASE_LINES = ["HEII4686", "NIIItrip", "CIII4650"]
FIT_STAGE_CHOICES = ["standard", "balmer_subtract"]
REGISTRY_CHECKPOINT_FIELDS = 100
REGISTRY_RECOVERY_PROGRESS_FILES = 100


def import_hotstars_normalisation(hotstars_code: Path):
    sys.path.insert(0, str(hotstars_code))
    from boss_normalisation import continuum_iterative_sigma_clip, TELLURIC_BANDS  # type: ignore

    return continuum_iterative_sigma_clip, TELLURIC_BANDS


def write_json_if_changed(path: Path, payload: dict[str, object]) -> bool:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return False
    path.write_text(rendered, encoding="utf-8")
    return True


def resolve_bundle_path(specfann_dir: Path, bundle_name: str | None, bundle_path: str | None) -> Path:
    if bundle_path:
        return Path(bundle_path).expanduser()
    return specfann_dir / "bundles" / (bundle_name or DEFAULT_BUNDLE_NAME)


def load_lines(lines: str | None, lines_file: str | None) -> list[str]:
    if lines_file:
        path = Path(lines_file)
        raw = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    if lines:
        return [item.strip() for item in lines.split(",") if item.strip()]
    return DEFAULT_LINES.copy()


def _coerce_sdss_id(value) -> str | None:
    if pd.isna(value):
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def load_manifest_selection(path: Path) -> dict[str, dict[str, str | None]]:
    df = pd.read_csv(path)
    required = {"field", "spec_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {', '.join(sorted(missing))}")
    fields = df["field"].astype(str).map(lambda val: val.zfill(6) if val.isdigit() else val)
    spec_files = df["spec_file"].astype(str)
    sdss_ids = None
    if "sdss_id" in df.columns:
        sdss_ids = df["sdss_id"].map(_coerce_sdss_id)
    selection: dict[str, dict[str, str | None]] = {}
    for idx, (field, spec_file) in enumerate(zip(fields, spec_files)):
        sdss_id = None if sdss_ids is None else sdss_ids.iloc[idx]
        selection.setdefault(field, {})[spec_file] = sdss_id
    return selection


def build_line_windows(
    bundle_path: Path,
    lines: list[str],
    *,
    window_pad: float = 0.0,
    rv_pad_kms: float = 0.0,
) -> tuple[dict[str, tuple[float, float]], list[str]]:
    windows: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for line in lines:
        wpath = bundle_path / f"wnew_{line}.npy"
        if not wpath.exists():
            missing.append(line)
            continue
        wave = np.load(wpath)
        if wave.size == 0:
            missing.append(line)
            continue
        lo = float(np.nanmin(wave))
        hi = float(np.nanmax(wave))
        pad_lo = lo * rv_pad_kms / C_KMS
        pad_hi = hi * rv_pad_kms / C_KMS
        lo = lo - window_pad - pad_lo
        hi = hi + window_pad + pad_hi
        windows[line] = (lo, hi)
    return windows, missing


def restrict_hdelta_group_windows(
    bundle_path: Path,
    windows: dict[str, tuple[float, float]],
    *,
    window_pad: float,
    rv_pad_kms: float,
) -> dict[str, tuple[float, float]]:
    if not any(line in windows for line in ("HDELTA", "HEI4143", "HEI4144")):
        return windows

    base_windows, _ = build_line_windows(
        bundle_path, HDELTA_GROUP_BASE_LINES, window_pad=window_pad, rv_pad_kms=rv_pad_kms
    )
    if not base_windows:
        return windows

    base_lo = min(lo for lo, _ in base_windows.values())
    base_hi = max(hi for _, hi in base_windows.values())

    restricted: dict[str, tuple[float, float]] = {}
    for line, (lo, hi) in windows.items():
        overlaps = not (hi < base_lo or lo > base_hi)
        if overlaps and not (lo >= base_lo and hi <= base_hi):
            continue
        restricted[line] = (lo, hi)
    return restricted


def group_windows(
    windows: dict[str, tuple[float, float]],
    *,
    merge_pad: float = 0.0,
) -> list[dict[str, object]]:
    items = sorted(windows.items(), key=lambda item: item[1][0])
    groups: list[dict[str, object]] = []
    for line, (lo, hi) in items:
        if not groups or lo > groups[-1]["hi"] + merge_pad:
            groups.append({"lo": lo, "hi": hi, "lines": [line]})
            continue
        groups[-1]["hi"] = max(groups[-1]["hi"], hi)
        groups[-1]["lines"].append(line)
    return groups


def build_groups(
    windows: dict[str, tuple[float, float]],
    *,
    merge_pad: float = 0.0,
    merge_windows: bool = True,
) -> list[dict[str, object]]:
    if merge_windows:
        remaining = dict(windows)
        groups: list[dict[str, object]] = []
        manual_groups = [
            HDELTA_GROUP_BASE_LINES,
            HGAMMA_GROUP_BASE_LINES,
            HEI4471_GROUP_BASE_LINES,
            HEII4686_GROUP_BASE_LINES,
        ]
        for group_lines in manual_groups:
            present = [line for line in group_lines if line in remaining]
            if not present:
                continue
            lo = min(remaining[line][0] for line in present)
            hi = max(remaining[line][1] for line in present)
            groups.append({"lo": lo, "hi": hi, "lines": present})
            for line in present:
                remaining.pop(line, None)
        groups.extend(group_windows(remaining, merge_pad=merge_pad))
        groups.sort(key=lambda item: item["lo"])
        return groups
    items = sorted(windows.items(), key=lambda item: item[1][0])
    return [{"lo": lo, "hi": hi, "lines": [line]} for line, (lo, hi) in items]


def line_title_color(line_name: str) -> str:
    name = line_name.upper()
    if name.startswith("HEII"):
        return "orange"
    if name.startswith("HEI"):
        return "red"
    if name.startswith("H"):
        return "black"
    if name.startswith("C"):
        return "green"
    if name.startswith("N"):
        return "blue"
    return "black"


def estimate_fwhm_from_norm(
    wave: np.ndarray,
    flux_norm: np.ndarray,
    fit_range: tuple[float, float],
    *,
    centre_hint: float | None = None,
    smooth_sigma_px: float = 0.0,
    eps: float = 1e-3,
) -> tuple[float | None, float | None, str | None]:
    lo, hi = fit_range
    mask = (wave >= lo) & (wave <= hi) & np.isfinite(wave) & np.isfinite(flux_norm)
    if mask.sum() < 3:
        return None, None, None
    w = wave[mask]
    f = flux_norm[mask]
    if smooth_sigma_px and f.size > 3:
        f = gaussian_filter1d(f, sigma=float(smooth_sigma_px), mode="nearest")

    f_min = float(np.min(f))
    f_max = float(np.max(f))
    if f_min < 1.0 - eps:
        half_level = 1.0 - 0.5 * (1.0 - f_min)
        select = f <= half_level
        centre = w[int(np.argmin(f))]
        kind = "absorption"
    elif f_max > 1.0 + eps:
        half_level = 1.0 + 0.5 * (f_max - 1.0)
        select = f >= half_level
        centre = w[int(np.argmax(f))]
        kind = "emission"
    else:
        return None, None, None

    if centre_hint is not None and np.isfinite(centre_hint):
        centre = centre_hint

    if select.sum() < 2:
        return None, None, kind

    centre_idx = int(np.argmin(np.abs(w - centre)))
    if not select[centre_idx]:
        sel_idx = np.where(select)[0]
        if sel_idx.size == 0:
            return None, None, kind
        centre_idx = int(sel_idx[np.argmin(np.abs(sel_idx - centre_idx))])

    left = centre_idx
    while left - 1 >= 0 and select[left - 1]:
        left -= 1
    right = centre_idx
    while right + 1 < select.size and select[right + 1]:
        right += 1

    fwhm = float(w[right] - w[left])
    if not np.isfinite(fwhm) or fwhm <= 0:
        return None, None, kind
    return fwhm, float(w[centre_idx]), kind


def compute_balmer_fwhm_masks(
    window_results: list[dict[str, object]],
    line_windows: dict[str, tuple[float, float]],
    *,
    k: float,
    smooth_sigma_px: float,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, object]]]:
    masks: dict[int, np.ndarray] = {}
    meta: dict[int, dict[str, object]] = {}

    for result in window_results:
        lines = [str(item) for item in result.get("lines", [])]
        balmer_lines = [line for line in BALMER_LINES if line in lines]
        if not balmer_lines:
            continue

        wave = np.asarray(result["wave"], dtype=float)
        flux_norm = np.asarray(result["flux_norm"], dtype=float)
        keep = np.ones_like(wave, dtype=bool)
        line_meta: dict[str, dict[str, object]] = {}

        for line in balmer_lines:
            fit_range = line_windows.get(line)
            if fit_range is None:
                continue
            lo, hi = fit_range
            lo = max(lo, float(np.nanmin(wave)))
            hi = min(hi, float(np.nanmax(wave)))
            if lo >= hi:
                continue
            centre_hint = BALMER_CENTRES.get(line)
            fwhm, centre, kind = estimate_fwhm_from_norm(
                wave,
                flux_norm,
                (lo, hi),
                centre_hint=centre_hint,
                smooth_sigma_px=smooth_sigma_px,
            )
            if fwhm is None or centre is None:
                continue
            half_width = 0.5 * float(k) * float(fwhm)
            mask_lo = centre - half_width
            mask_hi = centre + half_width
            keep &= ~((wave >= mask_lo) & (wave <= mask_hi))
            line_meta[line] = {
                "centre": float(centre),
                "fwhm": float(fwhm),
                "half_width": float(half_width),
                "mask_lo": float(mask_lo),
                "mask_hi": float(mask_hi),
                "fit_range": [float(lo), float(hi)],
                "kind": kind,
            }

        if line_meta:
            idx = int(result["group"])
            masks[idx] = keep
            meta[idx] = {"k": float(k), "smooth_sigma_px": float(smooth_sigma_px), "lines": line_meta}

    return masks, meta


def window_group_name(lines: list[str], idx: int, used_names: set[str]) -> str:
    if len(lines) == 1:
        base = lines[0]
    else:
        base = f"group_{idx:03d}"
    name = base
    suffix = 1
    while name in used_names:
        name = f"{base}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name


def load_spectrum(fits_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        if "LOGLAM" in data.columns.names:
            wave = 10 ** data["LOGLAM"].astype(float)
        else:
            wave = data["WAVELENGTH"].astype(float)
        flux = data["FLUX"].astype(float)
        ivar = data["IVAR"].astype(float) if "IVAR" in data.columns.names else None
    return wave, flux, ivar


def local_normalise_windows(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray | None,
    *,
    groups: list[dict[str, object]],
    line_windows: dict[str, tuple[float, float]] | None = None,
    continuum_iterative_sigma_clip,
    base_mask: np.ndarray,
    min_points: int,
    win_kms: float,
    min_win_px: int,
    sigma_lower: float,
    sigma_upper: float,
    n_iter: int,
    spline_order: int,
    smooth_scale: float,
    use_mad: bool,
    telluric_masks: list[tuple[float, float]] | None,
    eps: float,
    balmer_mask: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    used_names: set[str] = set()

    def fit_windows(
        *,
        mask_overrides: dict[int, np.ndarray] | None = None,
        fwhm_meta: dict[int, dict[str, object]] | None = None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        window_results: list[dict[str, object]] = []
        group_meta: list[dict[str, object]] = []

        for idx, group in enumerate(groups):
            lo = float(group["lo"])
            hi = float(group["hi"])
            lines = list(group["lines"])

            window_mask = (wave >= lo) & (wave <= hi)
            base_window_mask = base_mask & window_mask
            n_points = int(base_window_mask.sum())
            if n_points < min_points:
                group_meta.append(
                    {"group": idx, "lo": lo, "hi": hi, "lines": lines, "n_points": n_points, "used": False}
                )
                continue

            wave_win = wave[window_mask]
            flux_win = flux[window_mask]
            ivar_win = ivar[window_mask] if ivar is not None else None
            base_mask_win = base_mask[window_mask]
            fit_mask_win = base_mask_win
            mask_applied = False
            if mask_overrides is not None and idx in mask_overrides:
                override = mask_overrides[idx]
                if override.shape == base_mask_win.shape:
                    candidate = base_mask_win & override
                    if int(candidate.sum()) >= min_points:
                        fit_mask_win = candidate
                        mask_applied = True

            cont, meta = continuum_iterative_sigma_clip(
                wave_win,
                flux_win,
                base_mask=fit_mask_win,
                win_kms=win_kms,
                min_win_px=min_win_px,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                n_iter=n_iter,
                spline_order=spline_order,
                smooth_scale=smooth_scale,
                adaptive_smooth=False,
                use_mad=use_mad,
                telluric_masks=telluric_masks,
                eps=eps,
            )
            cont = np.clip(cont, eps, None)
            flux_norm = flux_win / cont

            ivar_norm = None
            if ivar_win is not None:
                ivar_norm = np.zeros_like(ivar_win, dtype=float)
                mpos = ivar_win > 0
                ivar_norm[mpos] = ivar_win[mpos] * (cont[mpos] ** 2)

            name = window_group_name(lines, idx, used_names)
            meta.update(
                {
                    "name": name,
                    "group": idx,
                    "lo": lo,
                    "hi": hi,
                    "lines": lines,
                    "n_points": n_points,
                    "used": True,
                }
            )
            if fwhm_meta is not None and idx in fwhm_meta:
                meta["balmer_fwhm"] = fwhm_meta[idx]
                meta["balmer_mask_applied"] = mask_applied
            group_meta.append(meta)
            window_results.append(
                {
                    "name": name,
                    "group": idx,
                    "lines": lines,
                    "lo": lo,
                    "hi": hi,
                    "n_points": n_points,
                    "wave": wave_win,
                    "flux": flux_win,
                    "ivar": ivar_win,
                    "flux_norm": flux_norm,
                    "ivar_norm": ivar_norm,
                    "continuum": cont,
                    "meta": meta,
                    "base_mask": base_mask_win,
                    "fit_mask": fit_mask_win,
                }
            )

        return window_results, group_meta

    if not balmer_mask or line_windows is None:
        return fit_windows()

    k = float(balmer_mask.get("k", 0.0))
    if k <= 0:
        return fit_windows()

    smooth_sigma_px = float(balmer_mask.get("smooth_sigma_px", 0.0))
    stage = str(balmer_mask.get("stage", "fit")).lower()

    prelim_results, prelim_meta = fit_windows()
    mask_overrides, fwhm_meta = compute_balmer_fwhm_masks(
        prelim_results,
        line_windows,
        k=k,
        smooth_sigma_px=smooth_sigma_px,
    )

    if stage in {"fit", "fit+p98"} and mask_overrides:
        window_results, group_meta = fit_windows(mask_overrides=mask_overrides, fwhm_meta=fwhm_meta)
    else:
        window_results, group_meta = prelim_results, prelim_meta
        if fwhm_meta:
            for result in window_results:
                idx = int(result["group"])
                if idx in fwhm_meta:
                    result["meta"]["balmer_fwhm"] = fwhm_meta[idx]

    for result in window_results:
        idx = int(result["group"])
        if idx in mask_overrides:
            result["fwhm_mask"] = mask_overrides[idx]
            result["fwhm_meta"] = fwhm_meta.get(idx)

    return window_results, group_meta


def local_normalise_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray | None,
    *,
    groups: list[dict[str, object]],
    continuum_iterative_sigma_clip,
    base_mask: np.ndarray,
    min_points: int,
    win_kms: float,
    min_win_px: int,
    sigma_lower: float,
    sigma_upper: float,
    n_iter: int,
    spline_order: int,
    smooth_scale: float,
    use_mad: bool,
    telluric_masks: list[tuple[float, float]] | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[dict[str, object]]]:
    continuum = np.ones_like(flux, dtype=float)
    group_meta: list[dict[str, object]] = []

    for idx, group in enumerate(groups):
        lo = float(group["lo"])
        hi = float(group["hi"])
        lines = list(group["lines"])

        mask = base_mask & (wave >= lo) & (wave <= hi)
        n_points = int(mask.sum())
        if n_points < min_points:
            group_meta.append(
                {"group": idx, "lo": lo, "hi": hi, "lines": lines, "n_points": n_points, "used": False}
            )
            continue

        cont, meta = continuum_iterative_sigma_clip(
            wave[mask],
            flux[mask],
            base_mask=base_mask[mask],
            win_kms=win_kms,
            min_win_px=min_win_px,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            n_iter=n_iter,
            spline_order=spline_order,
            smooth_scale=smooth_scale,
            adaptive_smooth=False,
            use_mad=use_mad,
            telluric_masks=telluric_masks,
            eps=eps,
        )
        continuum[mask] = cont
        meta.update({"group": idx, "lo": lo, "hi": hi, "lines": lines, "n_points": n_points, "used": True})
        group_meta.append(meta)

    flux_norm = flux / np.clip(continuum, eps, None)

    ivar_norm = None
    if ivar is not None:
        ivar_norm = np.zeros_like(ivar, dtype=float)
        mpos = ivar > 0
        ivar_norm[mpos] = ivar[mpos] * (continuum[mpos] ** 2)

    return flux_norm, ivar_norm, continuum, group_meta


def renorm_p98_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    *,
    base_mask: np.ndarray,
    line_range: tuple[float, float],
    renorm_percentile: float,
    spike_sigma: float,
    smooth_sigma_px: float,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    lo, hi = line_range
    in_range = (wave >= lo) & (wave <= hi)
    sample_mask = base_mask & in_range
    if extra_mask is not None and extra_mask.shape == sample_mask.shape:
        sample_mask &= extra_mask
    ratio = flux[sample_mask] / np.clip(continuum[sample_mask], 1e-12, None)
    ratio = ratio[np.isfinite(ratio)]
    if smooth_sigma_px and ratio.size > 3:
        ratio = gaussian_filter1d(ratio, sigma=float(smooth_sigma_px), mode="nearest")
    if ratio.size:
        med = np.nanmedian(ratio)
        mad = np.nanmedian(np.abs(ratio - med))
        sigma = 1.4826 * mad if mad > 0 else np.nanstd(ratio)
        if np.isfinite(sigma) and sigma > 0:
            ratio = ratio[ratio <= (med + spike_sigma * sigma)]
    scale = np.nanpercentile(ratio, renorm_percentile) if ratio.size else np.nan
    cont = continuum.copy()
    if np.isfinite(scale) and scale > 0:
        # Apply the scalar renorm consistently across the whole line range to
        # avoid artificial steps where base_mask/extra_mask boundaries fall.
        apply_mask = in_range & np.isfinite(continuum)
        cont[apply_mask] = continuum[apply_mask] * scale
    return cont


def apply_p98_renorm(
    wave: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    *,
    base_mask: np.ndarray,
    groups: list[dict[str, object]],
    renorm_percentile: float,
    spike_sigma: float,
    smooth_sigma_px: float,
    extra_masks: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    baseline = continuum
    renormed = baseline.copy()
    for idx, group in enumerate(groups):
        lo = float(group["lo"])
        hi = float(group["hi"])
        extra_mask = None
        if extra_masks is not None and idx in extra_masks:
            extra_mask = extra_masks[idx]
        cont_group = renorm_p98_continuum(
            wave,
            flux,
            baseline,
            base_mask=base_mask,
            line_range=(lo, hi),
            renorm_percentile=renorm_percentile,
            spike_sigma=spike_sigma,
            smooth_sigma_px=smooth_sigma_px,
            extra_mask=extra_mask,
        )
        mask = base_mask & (wave >= lo) & (wave <= hi)
        renormed[mask] = cont_group[mask]
    return renormed


class LocalBOSSSpectraProcessor:
    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        groups: list[dict[str, object]],
        line_windows: dict[str, tuple[float, float]],
        continuum_iterative_sigma_clip,
        telluric_masks: list[tuple[float, float]] | None,
        win_kms: float,
        min_win_px: int,
        sigma_lower: float,
        sigma_upper: float,
        n_iter: int,
        spline_order: int,
        smooth_scale: float,
        min_points: int,
        use_mad: bool,
        p98_renorm: bool,
        renorm_percentile: float,
        spike_sigma: float,
        smooth_sigma_px: float,
        store_continuum: bool,
        max_spectra: int | None,
        max_per_field: int | None,
        allowed_spectra: dict[str, dict[str, str | None]] | None,
        group_name_mode: str,
        plot_dir: Path | None,
        plot_lines: list[str] | None,
        write_output: bool,
        overwrite: bool,
        merge_windows: bool,
        balmer_mask_k: float | None,
        balmer_mask_stage: str,
        balmer_mask_smooth: float,
        fit_stage: str,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.groups = groups
        self.line_windows = line_windows
        self.continuum_iterative_sigma_clip = continuum_iterative_sigma_clip
        self.telluric_masks = telluric_masks
        self.win_kms = win_kms
        self.min_win_px = min_win_px
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.n_iter = n_iter
        self.spline_order = spline_order
        self.smooth_scale = smooth_scale
        self.min_points = min_points
        self.use_mad = use_mad
        self.p98_renorm = p98_renorm
        self.renorm_percentile = renorm_percentile
        self.spike_sigma = spike_sigma
        self.smooth_sigma_px = smooth_sigma_px
        self.store_continuum = store_continuum
        self.max_spectra = max_spectra
        self.max_per_field = max_per_field
        self.allowed_spectra = allowed_spectra
        self.group_name_mode = group_name_mode
        self.plot_dir = plot_dir
        self.plot_lines = plot_lines
        self.write_output = write_output
        self.overwrite = overwrite
        self.balmer_mask_k = balmer_mask_k
        self.balmer_mask_stage = balmer_mask_stage
        self.balmer_mask_smooth = balmer_mask_smooth
        self.fit_stage = str(fit_stage).lower()
        if self.fit_stage not in FIT_STAGE_CHOICES:
            raise ValueError(f"Unknown fit stage: {fit_stage!r}")
        if self.p98_renorm:
            self.normalisation_label = "local_line_windows_p98"
        else:
            self.normalisation_label = "local_line_windows"
        if self.fit_stage != "standard":
            self.normalisation_label = f"{self.normalisation_label}_{self.fit_stage}"
        if not merge_windows:
            self.normalisation_label = f"{self.normalisation_label}_per_line"

        if self.write_output:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.registry_file = self.output_dir / "processed_registry.parquet"
            if self.registry_file.exists():
                self.registry = pd.read_parquet(self.registry_file)
            else:
                self.registry = pd.DataFrame()
            self._prepare_registry()
            if not self.overwrite:
                self.recover_registry_from_outputs()
        else:
            self.registry_file = self.output_dir / "processed_registry.parquet"
            self.registry = pd.DataFrame()

    def _prepare_registry(self) -> None:
        required = [
            "field",
            "spectrum_name",
            "spec_file",
            "sdss_id",
            "processed_time",
            "h5_file",
            "halpha_emission_detected",
            "p98_emission_skip_lines_json",
        ]
        for col in required:
            if col not in self.registry.columns:
                self.registry[col] = pd.NA
        if self.registry["spec_file"].isna().all() and self.registry["spectrum_name"].notna().any():
            def infer_spec_file(name: object) -> str | None:
                if not isinstance(name, str):
                    return None
                if "spec-" in name:
                    return f"{name[name.find('spec-'): ]}.fits"
                return None
            self.registry["spec_file"] = self.registry["spectrum_name"].map(infer_spec_file)

    def _existing_registry_lookup(self) -> dict[str, set[str]]:
        lookup: dict[str, set[str]] = {}
        if self.registry.empty:
            return lookup
        grouped = self.registry.dropna(subset=["field", "spec_file"]).groupby("field")["spec_file"]
        for field, spec_files in grouped:
            lookup[str(field)] = set(spec_files.astype(str).tolist())
        return lookup

    def save_registry(self) -> None:
        self.registry.to_parquet(self.registry_file, index=False)

    def recover_registry_from_outputs(self) -> None:
        h5_paths = sorted(self.output_dir.glob("*.h5"))
        if not h5_paths:
            print("Registry recovery skipped: no existing field HDF5 outputs found.", flush=True)
            return

        print(
            f"Registry recovery starting: scanning {len(h5_paths)} existing field HDF5 files from {self.output_dir}",
            flush=True,
        )
        recovered_rows: list[dict[str, object]] = []
        bad_files: list[tuple[str, str]] = []
        for idx, h5_path in enumerate(h5_paths, start=1):
            try:
                with h5py.File(h5_path, "r") as handle:
                    for group_name in handle.keys():
                        grp = handle[group_name]
                        spec_file = grp.attrs.get("spec_file")
                        if isinstance(spec_file, bytes):
                            spec_file = spec_file.decode()
                        sdss_id = grp.attrs.get("sdss_id")
                        if isinstance(sdss_id, bytes):
                            sdss_id = sdss_id.decode()
                        recovered_rows.append(
                            {
                                "field": h5_path.stem,
                                "spectrum_name": str(group_name),
                                "spec_file": spec_file,
                                "sdss_id": None if sdss_id in (None, "") else str(sdss_id),
                                "processed_time": float(grp.attrs.get("processed_time", float("nan"))),
                                "h5_file": h5_path.name,
                                "halpha_emission_detected": bool(grp.attrs.get("halpha_emission_detected", False)),
                                "p98_emission_skip_lines_json": str(
                                    grp.attrs.get("p98_emission_skip_lines_json", "[]")
                                ),
                            }
                        )
            except Exception as exc:
                bad_files.append((h5_path.name, repr(exc)))
                print(f"[corrupt-output] path={h5_path} reason={exc!r}", flush=True)
            if idx % REGISTRY_RECOVERY_PROGRESS_FILES == 0 or idx == len(h5_paths):
                print(
                    "Registry recovery progress: "
                    f"scanned_h5_files={idx}/{len(h5_paths)} "
                    f"recovered_rows={len(recovered_rows)} "
                    f"bad_h5_files={len(bad_files)}",
                    flush=True,
                )

        if recovered_rows:
            before = len(self.registry.drop_duplicates(subset=["field", "spec_file"], keep="last"))
            self.registry = pd.concat([self.registry, pd.DataFrame(recovered_rows)], ignore_index=True)
            self.registry = self.registry.drop_duplicates(
                subset=["field", "spec_file"],
                keep="last",
            ).reset_index(drop=True)
            after = len(self.registry)
            if after != before:
                self.save_registry()
            print(
                f"Registry recovery complete: added_rows={after - before} "
                f"total_rows={after} scanned_h5_files={len(h5_paths)} bad_h5_files={len(bad_files)}",
                flush=True,
            )
        else:
            print(
                f"Registry recovery complete: added_rows=0 total_rows={len(self.registry)} "
                f"scanned_h5_files={len(h5_paths)} bad_h5_files={len(bad_files)}",
                flush=True,
            )
        if bad_files:
            preview = "; ".join(f"{name}: {reason}" for name, reason in bad_files[:5])
            print(
                f"[corrupt-output-summary] bad_h5_files={len(bad_files)} preview={preview}",
                flush=True,
            )

    def flush_registry_updates(self, registry_updates: list[pd.DataFrame]) -> int:
        if not registry_updates:
            return 0
        added_rows = int(sum(len(df) for df in registry_updates))
        self.registry = pd.concat([self.registry, *registry_updates], ignore_index=True)
        self.registry = self.registry.drop_duplicates(
            subset=["field", "spec_file"],
            keep="last",
        ).reset_index(drop=True)
        self.save_registry()
        registry_updates.clear()
        return added_rows

    def process_spectrum(
        self, fits_path: Path
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
        wave, flux, ivar = load_spectrum(fits_path)
        base_mask = np.isfinite(wave) & np.isfinite(flux)
        if ivar is not None:
            base_mask &= np.isfinite(ivar) & (ivar > 0)
        if base_mask.sum() < self.min_points:
            raise RuntimeError(f"Too few valid points in {fits_path.name}")

        balmer_mask = None
        if self.balmer_mask_k is not None and self.balmer_mask_k > 0:
            balmer_mask = {
                "k": float(self.balmer_mask_k),
                "stage": str(self.balmer_mask_stage),
                "smooth_sigma_px": float(self.balmer_mask_smooth),
            }

        window_results, group_meta = local_normalise_windows(
            wave,
            flux,
            ivar,
            groups=self.groups,
            line_windows=self.line_windows,
            continuum_iterative_sigma_clip=self.continuum_iterative_sigma_clip,
            base_mask=base_mask,
            min_points=self.min_points,
            win_kms=self.win_kms,
            min_win_px=self.min_win_px,
            sigma_lower=self.sigma_lower,
            sigma_upper=self.sigma_upper,
            n_iter=self.n_iter,
            spline_order=self.spline_order,
            smooth_scale=self.smooth_scale,
            use_mad=self.use_mad,
            telluric_masks=self.telluric_masks,
            eps=1e-12,
            balmer_mask=balmer_mask,
        )
        if self.fit_stage != "standard":
            bs_kwargs = {}
            if self.fit_stage == "balmer_subtract":
                bs_kwargs["continuum_fit_func"] = self.continuum_iterative_sigma_clip
                bs_kwargs["continuum_fit_kwargs"] = dict(
                    win_kms=self.win_kms,
                    min_win_px=self.min_win_px,
                    sigma_lower=self.sigma_lower,
                    sigma_upper=self.sigma_upper,
                    n_iter=self.n_iter,
                    spline_order=self.spline_order,
                    smooth_scale=self.smooth_scale,
                    use_mad=self.use_mad,
                    telluric_masks=self.telluric_masks,
                    eps=1e-12,
                )
                bs_kwargs["use_fwhm_mask"] = self.balmer_mask_stage in {"fit", "fit+p98"}
            apply_fitstage_to_balmer_windows(
                self.fit_stage,
                window_results,
                self.line_windows,
                **bs_kwargs,
            )
        p98_emission_skip_lines: list[str] = []
        if self.p98_renorm:
            for result in window_results:
                wave_win = result["wave"]
                flux_win = result["flux"]
                cont = result["continuum"]
                base_mask_win = result["base_mask"]
                extra_mask = None
                if balmer_mask is not None and self.balmer_mask_stage in {"p98", "fit+p98"}:
                    extra_mask = result.get("fwhm_mask")
                # For Balmer windows where the current flux_norm shows emission
                # near a Balmer line centre (median within 5 Å of centre > 1.05),
                # skip p98 entirely.  The emission peak biases the percentile
                # upward, scaling the continuum to match the peak height and
                # suppressing the emission in the normalised spectrum.  Applying
                # fwhm_mask as extra_mask is not used here because the emission
                # FWHM mask can be very wide for broad emission lines, leaving
                # too few pseudo-continuum pixels for a reliable scale estimate.
                # The balmer_subtract refit (sigma_upper clipping of the spline)
                # already places the continuum at the pseudo-continuum level.
                if extra_mask is None:
                    _wave_w = np.asarray(result["wave"], dtype=float)
                    _fn_w = np.asarray(result.get("flux_norm", []), dtype=float)
                    if _fn_w.size == _wave_w.size:
                        _group_lines = [str(_l) for _l in result.get("lines", [])]
                        for _line in BALMER_LINES:
                            if _line not in _group_lines:
                                continue
                            _centre = BALMER_CENTRES.get(_line)
                            if _centre is None:
                                continue
                            _near = (np.abs(_wave_w - _centre) < 5.0) & np.isfinite(_fn_w)
                            if _near.sum() >= 3 and float(np.nanmedian(_fn_w[_near])) > 1.05:
                                break  # emission detected — skip p98 for this window
                        else:
                            _line = None  # no emission found, proceed normally
                        if _line is not None:
                            if _line not in p98_emission_skip_lines:
                                p98_emission_skip_lines.append(_line)
                            result["meta"]["p98_emission_skip_line"] = _line
                            continue  # skip p98 renorm for this emission window
                cont = renorm_p98_continuum(
                    wave_win,
                    flux_win,
                    cont,
                    base_mask=base_mask_win,
                    line_range=(result["lo"], result["hi"]),
                    renorm_percentile=self.renorm_percentile,
                    spike_sigma=self.spike_sigma,
                    smooth_sigma_px=self.smooth_sigma_px,
                    extra_mask=extra_mask,
                )
                result["continuum"] = cont
                flux_norm = flux_win / np.clip(cont, 1e-12, None)
                result["flux_norm"] = flux_norm
                ivar_win = result["ivar"]
                if ivar_win is not None:
                    ivar_norm = np.zeros_like(ivar_win, dtype=float)
                    mpos = ivar_win > 0
                    ivar_norm[mpos] = ivar_win[mpos] * (cont[mpos] ** 2)
                    result["ivar_norm"] = ivar_norm
        return window_results, group_meta, p98_emission_skip_lines

    def plot_line_diagnostics(
        self,
        *,
        spec_name: str,
        window_result: dict[str, object],
    ) -> None:
        if self.plot_dir is None:
            return

        lines = [str(item) for item in window_result["lines"]]
        if self.plot_lines is not None and not any(line in self.plot_lines for line in lines):
            return

        wave = window_result["wave"]
        flux = window_result["flux"]
        flux_norm = window_result["flux_norm"]
        continuum = window_result["continuum"]

        out_root = self.plot_dir / spec_name
        out_root.mkdir(parents=True, exist_ok=True)

        if wave.size < 5:
            return

        fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.0), sharex=True)
        ax0, ax1 = axes
        ax0.plot(wave, flux, color="k", lw=1.0, label="Flux")
        ax0.plot(wave, continuum, color="tab:orange", lw=1.6, label="Continuum")
        ax0.set_ylabel("Flux")
        ax0.legend(loc="best", fontsize=8)

        ax1.plot(wave, flux_norm, color="tab:red", lw=1.2, label="Norm flux")
        ax1.axhline(1.0, ls="--", color="gray", alpha=0.6)
        ax1.set_ylabel("Norm flux")
        ax1.set_xlabel("Wavelength (A)")

        label = "+".join(lines)
        title = f"{spec_name} — {label}"
        ax0.set_title(title, color=line_title_color(lines[0]))
        fig.tight_layout()

        out_path = out_root / f"{window_result['name']}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def process_field(
        self,
        field_dir: Path,
        *,
        existing_spec_files: set[str] | None = None,
        remaining_total: int | None = None,
    ) -> dict[str, object]:
        field_name = field_dir.name
        if self.allowed_spectra is not None and field_name not in self.allowed_spectra:
            return {
                "field": field_name,
                "processed_count": 0,
                "failed_count": 0,
                "updates": pd.DataFrame(
                    columns=[
                        "field",
                        "spectrum_name",
                        "spec_file",
                        "sdss_id",
                        "processed_time",
                        "h5_file",
                        "halpha_emission_detected",
                        "p98_emission_skip_lines_json",
                    ]
                ),
            }
        output_file = self.output_dir / f"{field_name}.h5"
        processed_here = 0
        failed_here = 0
        new_rows: list[dict[str, object]] = []
        existing_spec_files = existing_spec_files or set()

        fits_paths = []
        if self.allowed_spectra is None:
            fits_paths = sorted(field_dir.glob("*.fits"))
        else:
            for spec_file in sorted(self.allowed_spectra[field_name].keys()):
                fits_path = field_dir / spec_file
                if fits_path.exists():
                    fits_paths.append(fits_path)
                else:
                    print(f"[missing] {fits_path}")

        h5f = None
        if self.write_output:
            if not self.overwrite and fits_paths and all(path.name in existing_spec_files for path in fits_paths):
                updates = pd.DataFrame(
                    columns=[
                        "field",
                        "spectrum_name",
                        "spec_file",
                        "sdss_id",
                        "processed_time",
                        "h5_file",
                        "halpha_emission_detected",
                        "p98_emission_skip_lines_json",
                    ]
                )
                return {
                    "field": field_name,
                    "processed_count": 0,
                    "failed_count": 0,
                    "updates": updates,
                }
            open_mode = "a"
            try:
                h5f = h5py.File(output_file, open_mode)
            except Exception as exc:
                if output_file.exists() and not self.overwrite and not existing_spec_files:
                    print(
                        f"[corrupt-output] field={field_name} path={output_file} "
                        f"action=recreate reason={exc}",
                        flush=True,
                    )
                    open_mode = "w"
                    h5f = h5py.File(output_file, open_mode)
                else:
                    raise RuntimeError(f"Failed to open output file {output_file}: {exc}") from exc
            if "normalisation" not in h5f.attrs:
                h5f.attrs["normalisation"] = self.normalisation_label
            h5f.attrs["grid"] = "windowed"
            h5f.attrs["output_format"] = "windowed"

            for fits_path in fits_paths:
                if remaining_total is not None and processed_here >= remaining_total:
                    break
                if self.max_per_field is not None and processed_here >= self.max_per_field:
                    break

                spec_name = fits_path.stem
                spec_file = fits_path.name
                sdss_id = None
                if self.allowed_spectra is not None:
                    sdss_id = self.allowed_spectra[field_name].get(spec_file)

                if self.group_name_mode == "sdssid_specfile" and sdss_id:
                    group_name = f"{sdss_id}__{spec_name}"
                elif self.group_name_mode == "sdssid" and sdss_id:
                    group_name = sdss_id
                else:
                    group_name = spec_name

                if spec_file in existing_spec_files and not self.overwrite:
                    continue
                if h5f is not None and group_name in h5f and not self.overwrite:
                    continue

                try:
                    window_results, group_meta, p98_emission_skip_lines = self.process_spectrum(fits_path)
                    halpha_emission_detected = "HALPHA" in p98_emission_skip_lines
                    if p98_emission_skip_lines:
                        skip_lines_str = ",".join(p98_emission_skip_lines)
                        sdss_id_str = str(sdss_id) if sdss_id is not None else "NA"
                        print(
                            "[balmer-emission] "
                            f"field={field_name} "
                            f"spec_file={spec_file} "
                            f"sdss_id={sdss_id_str} "
                            f"halpha_emission_detected={str(halpha_emission_detected).lower()} "
                            f"skipped_lines={skip_lines_str}"
                        )
                    if self.plot_dir is not None:
                        for result in window_results:
                            self.plot_line_diagnostics(spec_name=spec_name, window_result=result)

                    if self.write_output and h5f is not None:
                        if self.overwrite and group_name in h5f:
                            del h5f[group_name]
                        grp = h5f.create_group(group_name)
                        for result in window_results:
                            win_name = result["name"]
                            win = grp.create_group(win_name)
                            win.create_dataset("wave", data=result["wave"], compression="gzip", compression_opts=6)
                            win.create_dataset("flux", data=result["flux_norm"], compression="gzip", compression_opts=6)
                            if result["ivar_norm"] is not None:
                                win.create_dataset("ivar", data=result["ivar_norm"], compression="gzip", compression_opts=6)
                            if self.store_continuum:
                                win.create_dataset(
                                    "continuum", data=result["continuum"], compression="gzip", compression_opts=6
                                )
                            win.attrs["lines_json"] = json.dumps(result["lines"])
                            win.attrs["lo"] = result["lo"]
                            win.attrs["hi"] = result["hi"]
                            win.attrs["n_points"] = result["n_points"]
                            win.attrs["meta_json"] = json.dumps(result["meta"])
                        grp.attrs["original_file"] = str(fits_path)
                        grp.attrs["processed_time"] = time.time()
                        grp.attrs["normalisation"] = self.normalisation_label
                        grp.attrs["fit_stage"] = self.fit_stage
                        grp.attrs["output_format"] = "windowed"
                        grp.attrs["spec_file"] = spec_file
                        if sdss_id is not None:
                            grp.attrs["sdss_id"] = sdss_id
                        grp.attrs["halpha_emission_detected"] = halpha_emission_detected
                        grp.attrs["p98_emission_skip_lines_json"] = json.dumps(p98_emission_skip_lines)
                        grp.attrs["line_groups_json"] = json.dumps(group_meta)

                        new_rows.append(
                            {
                                "field": field_name,
                                "spectrum_name": group_name,
                                "spec_file": spec_file,
                                "sdss_id": sdss_id,
                                "processed_time": grp.attrs["processed_time"],
                                "h5_file": output_file.name,
                                "halpha_emission_detected": halpha_emission_detected,
                                "p98_emission_skip_lines_json": json.dumps(p98_emission_skip_lines),
                            }
                        )
                        processed_here += 1

                except Exception as exc:
                    failed_here += 1
                    print(f"Failed to process {fits_path}: {exc}")
        if h5f is not None:
            h5f.close()

        updates = pd.DataFrame(
            new_rows,
            columns=[
                "field",
                "spectrum_name",
                "spec_file",
                "sdss_id",
                "processed_time",
                "h5_file",
                "halpha_emission_detected",
                "p98_emission_skip_lines_json",
            ],
        )
        return {
            "field": field_name,
            "processed_count": processed_here,
            "failed_count": failed_here,
            "updates": updates,
        }

    def process_all_fields(self, field_filter: list[str] | None = None, max_workers: int = 1) -> None:
        if self.allowed_spectra is None:
            field_dirs = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
        else:
            field_dirs = sorted(
                [
                    self.input_dir / field
                    for field in sorted(self.allowed_spectra.keys())
                    if (self.input_dir / field).is_dir()
                ]
            )
        if field_filter:
            field_dirs = [field_dir for field_dir in field_dirs if field_dir.name in field_filter]

        total_fields = len(field_dirs)
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_workers > 1 and self.max_spectra is not None:
            print("Parallel mode disabled because --max-spectra requires an exact global cap; using 1 worker.")
            max_workers = 1
        if max_workers > 1 and self.plot_dir is not None:
            print("Parallel mode disabled because Matplotlib plotting is not thread-safe here; using 1 worker.")
            max_workers = 1

        existing_lookup = self._existing_registry_lookup()
        registry_updates: list[pd.DataFrame] = []
        processed_total = 0
        failed_spectra = 0
        completed_fields = 0
        failed_fields = 0

        print(f"Found {total_fields} fields")

        def collect_result(result: dict[str, object]) -> None:
            nonlocal processed_total, failed_spectra
            updates = result["updates"]
            if isinstance(updates, pd.DataFrame) and not updates.empty:
                registry_updates.append(updates)
            processed_total += int(result["processed_count"])
            failed_spectra += int(result["failed_count"])

        if max_workers == 1:
            for field_dir in field_dirs:
                if self.max_spectra is not None and processed_total >= self.max_spectra:
                    break
                remaining_total = None
                if self.max_spectra is not None:
                    remaining_total = self.max_spectra - processed_total
                try:
                    result = self.process_field(
                        field_dir,
                        existing_spec_files=existing_lookup.get(field_dir.name, set()),
                        remaining_total=remaining_total,
                    )
                    collect_result(result)
                    completed_fields += 1
                    print(
                        f"Completed field {field_dir.name}: "
                        f"{int(result['processed_count'])} spectra, {int(result['failed_count'])} failures",
                        flush=True,
                    )
                    if self.write_output and registry_updates and completed_fields % REGISTRY_CHECKPOINT_FIELDS == 0:
                        checkpoint_rows = self.flush_registry_updates(registry_updates)
                        print(
                            f"Registry checkpoint saved at {self.registry_file}: "
                            f"added_rows={checkpoint_rows} total_rows={len(self.registry)} "
                            f"completed_fields={completed_fields} spectra_processed={processed_total}",
                            flush=True,
                        )
                except Exception as exc:
                    failed_fields += 1
                    print(f"Failed field {field_dir.name}: {exc}", flush=True)
        else:
            print(f"Processing fields in parallel with {max_workers} workers", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_field = {
                    executor.submit(
                        self.process_field,
                        field_dir,
                        existing_spec_files=existing_lookup.get(field_dir.name, set()),
                    ): field_dir.name
                    for field_dir in field_dirs
                }
                for future in concurrent.futures.as_completed(future_to_field):
                    field_name = future_to_field[future]
                    try:
                        result = future.result()
                        collect_result(result)
                        completed_fields += 1
                        print(
                            f"Completed field {field_name}: "
                            f"{int(result['processed_count'])} spectra, {int(result['failed_count'])} failures",
                            flush=True,
                        )
                        if self.write_output and registry_updates and completed_fields % REGISTRY_CHECKPOINT_FIELDS == 0:
                            checkpoint_rows = self.flush_registry_updates(registry_updates)
                            print(
                                f"Registry checkpoint saved at {self.registry_file}: "
                                f"added_rows={checkpoint_rows} total_rows={len(self.registry)} "
                                f"completed_fields={completed_fields} spectra_processed={processed_total}",
                                flush=True,
                            )
                    except Exception as exc:
                        failed_fields += 1
                        print(f"Failed field {field_name}: {exc}", flush=True)

        if registry_updates:
            self.flush_registry_updates(registry_updates)
        registry_saved = False
        if self.write_output and not self.registry.empty:
            registry_saved = True

        print("\nProcessing summary:", flush=True)
        print(f"Fields scheduled: {total_fields}", flush=True)
        print(f"Fields completed: {completed_fields}", flush=True)
        print(f"Fields failed: {failed_fields}", flush=True)
        print(f"Spectra processed: {processed_total}", flush=True)
        print(f"Spectra failed: {failed_spectra}", flush=True)
        if processed_total == 0:
            print("No spectra were processed; existing outputs were left unchanged.", flush=True)
        elif self.write_output and not registry_saved:
            print("No registry updates were required.", flush=True)
        if failed_fields or failed_spectra:
            raise RuntimeError(
                "Local-normalisation run completed with failures: "
                f"failed_fields={failed_fields}, failed_spectra={failed_spectra}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local normalisation for BOSS spectra using line windows.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hotstars-code", type=Path, default=DEFAULT_HOTSTARS_CODE)
    parser.add_argument("--specfann-dir", type=Path, default=DEFAULT_SPECFANN_DIR)
    parser.add_argument("--bundle-name", type=str, default=DEFAULT_BUNDLE_NAME)
    parser.add_argument("--bundle-path", type=str, default=None)
    parser.add_argument("--lines", type=str, default=None, help="Comma-separated line list.")
    parser.add_argument("--lines-file", type=str, default=None, help="Text file with one line id per line.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest with columns field,spec_file to restrict processing.",
    )
    parser.add_argument(
        "--group-name",
        type=str,
        default="auto",
        choices=["auto", "specfile", "sdssid_specfile", "sdssid"],
        help="Naming scheme for HDF5 groups (auto uses sdssid+specfile if available).",
    )
    parser.add_argument("--window-pad", type=float, default=2.0, help="Extra padding (Å) around each line window.")
    parser.add_argument("--rv-pad-kms", type=float, default=500.0, help="Pad windows for RV shifts (km/s).")
    parser.add_argument("--merge-pad", type=float, default=0.0, help="Merge window gaps smaller than this (Å).")
    parser.add_argument(
        "--no-merge",
        dest="merge_windows",
        action="store_false",
        default=True,
        help="Do not merge overlapping line windows; treat each line independently.",
    )
    parser.add_argument("--min-points", type=int, default=40, help="Min points required in a window.")
    parser.add_argument("--win-kms", type=float, default=1200.0, help="Sigma-clip window size (km/s).")
    parser.add_argument("--min-win-px", type=int, default=21, help="Minimum window size in pixels.")
    parser.add_argument("--sigma-lower", type=float, default=2.0)
    parser.add_argument("--sigma-upper", type=float, default=3.0)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--smooth-scale", type=float, default=50.0, help="Spline knot spacing (Å).")
    parser.add_argument(
        "--use-mad",
        dest="use_mad",
        action="store_true",
        default=True,
        help="Use MAD-based sigma clipping (default: on).",
    )
    parser.add_argument(
        "--no-use-mad",
        dest="use_mad",
        action="store_false",
        help="Disable MAD-based sigma clipping.",
    )
    parser.add_argument(
        "--p98-renorm",
        dest="p98_renorm",
        action="store_true",
        default=True,
        help="Apply p98 continuum renormalisation in line windows (default: on).",
    )
    parser.add_argument(
        "--no-p98-renorm",
        dest="p98_renorm",
        action="store_false",
        help="Disable p98 continuum renormalisation.",
    )
    parser.add_argument(
        "--renorm-percentile",
        type=float,
        default=98.0,
        help="Percentile used for p98 renormalisation.",
    )
    parser.add_argument(
        "--spike-sigma",
        type=float,
        default=4.0,
        help="Sigma threshold for clipping high spikes before p98 renormalisation.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma (pixels) applied to ratio before p98 renormalisation.",
    )
    parser.add_argument("--mask-telluric", action="store_true", help="Mask common telluric/sky bands.")
    parser.add_argument(
        "--balmer-mask-k",
        type=float,
        default=None,
        help="Mask +/- k*FWHM around Balmer line cores when estimating the continuum.",
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
    parser.add_argument(
        "--fit-stage",
        type=str,
        default="standard",
        choices=FIT_STAGE_CHOICES,
        help="Balmer fit-stage strategy applied before optional p98 renorm.",
    )
    parser.add_argument("--store-continuum", action="store_true", help="Store continuum arrays in HDF5.")
    parser.add_argument("--max-spectra", type=int, default=None, help="Total spectra cap for quick tests.")
    parser.add_argument("--max-per-field", type=int, default=None, help="Per-field spectra cap.")
    parser.add_argument("--plot", action="store_true", help="Write per-line continuum diagnostics.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Output directory for line plots.")
    parser.add_argument(
        "--plot-lines",
        type=str,
        default=None,
        help="Comma-separated line list to plot (default: all line windows).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots; do not write HDF5 or update registry.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing groups and registry entries for matching spectra.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated field directory names to process (default: all).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of field directories to process in parallel.",
    )

    args = parser.parse_args()

    continuum_iterative_sigma_clip, telluric_bands = import_hotstars_normalisation(args.hotstars_code)

    allowed_spectra = None
    if args.manifest is not None:
        allowed_spectra = load_manifest_selection(args.manifest)

    group_name_mode = args.group_name
    if group_name_mode == "auto":
        has_sdssid = False
        if allowed_spectra is not None:
            for field_map in allowed_spectra.values():
                if any(value is not None for value in field_map.values()):
                    has_sdssid = True
                    break
        group_name_mode = "sdssid_specfile" if has_sdssid else "specfile"

    bundle_path = resolve_bundle_path(args.specfann_dir, args.bundle_name, args.bundle_path)
    lines = load_lines(args.lines, args.lines_file)

    windows, missing = build_line_windows(
        bundle_path,
        lines,
        window_pad=args.window_pad,
        rv_pad_kms=args.rv_pad_kms,
    )
    windows = restrict_hdelta_group_windows(
        bundle_path, windows, window_pad=args.window_pad, rv_pad_kms=args.rv_pad_kms
    )
    if missing:
        print(f"Missing line windows for {len(missing)} lines: {', '.join(missing)}")
    if not windows:
        raise SystemExit("No line windows available; check bundle path and line list.")

    groups = build_groups(windows, merge_pad=args.merge_pad, merge_windows=args.merge_windows)
    plot_lines = None
    if args.plot_lines:
        plot_lines = [item.strip() for item in args.plot_lines.split(",") if item.strip()]
    plot_dir = None
    if args.plot:
        plot_dir = args.plot_dir or (args.output_dir / "figures")

    config_path = args.output_dir / "local_normalisation_config.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "bundle_path": str(bundle_path),
        "lines": lines,
        "manifest": None if args.manifest is None else str(args.manifest),
        "group_name_mode": group_name_mode,
        "output_format": "windowed",
        "window_pad": args.window_pad,
        "rv_pad_kms": args.rv_pad_kms,
        "merge_pad": args.merge_pad,
        "merge_windows": args.merge_windows,
        "plot": bool(args.plot),
        "plot_dir": None if plot_dir is None else str(plot_dir),
        "plot_lines": plot_lines,
        "max_workers": args.max_workers,
        "groups": groups,
        "overwrite": args.overwrite,
        "normalisation": {
            "win_kms": args.win_kms,
            "min_win_px": args.min_win_px,
            "sigma_lower": args.sigma_lower,
            "sigma_upper": args.sigma_upper,
            "n_iter": args.n_iter,
            "spline_order": args.spline_order,
            "smooth_scale": args.smooth_scale,
            "adaptive_smooth": False,
            "use_mad": args.use_mad,
            "mask_telluric": args.mask_telluric,
            "p98_renorm": args.p98_renorm,
            "p98_scope": "window",
            "renorm_percentile": args.renorm_percentile,
            "spike_sigma": args.spike_sigma,
            "smooth_sigma_px": args.smooth_sigma,
            "fit_stage": args.fit_stage,
        },
    }
    if args.balmer_mask_k is not None and args.balmer_mask_k > 0:
        config_payload["normalisation"]["balmer_mask"] = {
            "k": args.balmer_mask_k,
            "stage": args.balmer_mask_stage,
            "smooth_sigma_px": args.balmer_mask_smooth,
        }
    config_written = write_json_if_changed(config_path, config_payload)

    telluric_masks = telluric_bands if args.mask_telluric else None
    field_filter = [item.strip() for item in args.fields.split(",") if item.strip()] if args.fields else None

    processor = LocalBOSSSpectraProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        groups=groups,
        line_windows=windows,
        continuum_iterative_sigma_clip=continuum_iterative_sigma_clip,
        telluric_masks=telluric_masks,
        win_kms=args.win_kms,
        min_win_px=args.min_win_px,
        sigma_lower=args.sigma_lower,
        sigma_upper=args.sigma_upper,
        n_iter=args.n_iter,
        spline_order=args.spline_order,
        smooth_scale=args.smooth_scale,
        min_points=args.min_points,
        use_mad=args.use_mad,
        p98_renorm=args.p98_renorm,
        renorm_percentile=args.renorm_percentile,
        spike_sigma=args.spike_sigma,
        smooth_sigma_px=args.smooth_sigma,
        store_continuum=args.store_continuum,
        max_spectra=args.max_spectra,
        max_per_field=args.max_per_field,
        allowed_spectra=allowed_spectra,
        group_name_mode=group_name_mode,
        plot_dir=plot_dir,
        plot_lines=plot_lines,
        write_output=not args.plot_only,
        overwrite=args.overwrite,
        merge_windows=args.merge_windows,
        balmer_mask_k=args.balmer_mask_k,
        balmer_mask_stage=args.balmer_mask_stage,
        balmer_mask_smooth=args.balmer_mask_smooth,
        fit_stage=args.fit_stage,
    )
    processor.process_all_fields(field_filter=field_filter, max_workers=args.max_workers)

    if processor.write_output and processor.registry_file.exists():
        print(f"Registry available at {processor.registry_file}")
    if config_written:
        print(f"Config written at {config_path}")
    else:
        print(f"Config unchanged at {config_path}")


if __name__ == "__main__":
    main()
