#!/usr/bin/env python3
"""
Alternative continuum-fitting strategies for Balmer windows.

These methods replace the default sigma-clip spline continuum with approaches
that are more robust to the broad Balmer wing absorption pull-down problem.

Methods implemented
-------------------
1. quantile_spline — Quantile regression spline (IRLS) targeting an upper
   quantile (e.g. tau=0.85).  Naturally biases the continuum upward toward
   the pseudo-continuum envelope.

2. penalised_upper — Iterative upper-envelope spline.  Repeatedly removes
   points below the current fit and refits, converging on the upper boundary
   of the flux distribution.

3. lorentzian_deblend — Joint Lorentzian absorption + linear continuum fit.
   Uses the fitted polynomial continuum component directly, cleanly
   separating the wing from the pseudo-continuum.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median_smooth(y: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-smooth with NaN-safe fill."""
    finite = np.isfinite(y)
    if np.count_nonzero(finite) < 3 or sigma_px <= 0:
        return y.copy()
    fill = float(np.nanmedian(y[finite]))
    work = y.copy()
    work[~finite] = fill
    out = gaussian_filter1d(work, sigma=sigma_px, mode="nearest")
    out[~finite] = np.nan
    return out


def _knot_positions(wave: np.ndarray, spacing: float) -> np.ndarray:
    """Return interior knot positions spaced by *spacing* Angstroms."""
    lo, hi = float(wave.min()), float(wave.max())
    if hi - lo <= 2.0 * spacing:
        return np.array([], dtype=float)
    knots = np.arange(lo + spacing, hi - spacing + 1e-6, spacing)
    return knots


# ---------------------------------------------------------------------------
# Method 1 — Quantile Regression Spline (IRLS)
# ---------------------------------------------------------------------------

def fit_quantile_spline(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    tau: float = 0.85,
    knot_spacing: float = 8.0,
    n_iter: int = 15,
    smooth_sigma_px: float = 1.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Fit a spline targeting the *tau*-th quantile of flux vs wavelength.

    Uses iteratively reweighted least squares (IRLS) with pinball-loss
    weights to approximate quantile regression with a smooth spline.

    Parameters
    ----------
    wave, flux : wavelength and un-normalised flux arrays (same length).
    tau : quantile level (0 < tau < 1).  tau > 0.5 biases the fit upward.
    knot_spacing : interior knot spacing in Angstroms.
    n_iter : maximum IRLS iterations.
    smooth_sigma_px : Gaussian smoothing applied to the initial flux estimate.
    eps : small constant to avoid zero weights.

    Returns
    -------
    continuum : array of the same shape as *flux*.
    """
    valid = np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(valid) < 10:
        return np.full_like(flux, np.nan, dtype=float)

    w = wave[valid]
    f = flux[valid]

    # Initial estimate: smoothed flux.
    cont = _median_smooth(f, smooth_sigma_px)

    knots = _knot_positions(w, knot_spacing)
    if knots.size < 1:
        # Window too narrow for spline — use a weighted polynomial.
        return _quantile_polyfit(wave, flux, valid, tau=tau, deg=2)

    for _ in range(n_iter):
        residual = f - cont
        # Pinball (check) loss weights:
        #   w_i = tau           if residual_i >= 0  (underestimate)
        #   w_i = (1 - tau)     if residual_i < 0   (overestimate)
        weights = np.where(residual >= 0, tau, 1.0 - tau)
        weights = np.clip(weights, eps, None)

        try:
            spl = UnivariateSpline(w, f, w=weights, t=knots, k=3, ext=0)
            cont = spl(w)
        except Exception:
            # Fallback: simple weighted polynomial.
            return _quantile_polyfit(wave, flux, valid, tau=tau, deg=2)

    out = np.full_like(flux, np.nan, dtype=float)
    out[valid] = cont
    return out


def _quantile_polyfit(
    wave: np.ndarray,
    flux: np.ndarray,
    valid: np.ndarray,
    *,
    tau: float,
    deg: int,
) -> np.ndarray:
    """Fallback quantile-like polynomial fit."""
    w = wave[valid]
    f = flux[valid]
    cont = np.polyval(np.polyfit(w, f, deg), w)
    for _ in range(8):
        residual = f - cont
        weights = np.where(residual >= 0, tau, 1.0 - tau)
        weights = np.clip(weights, 1e-8, None)
        cont = np.polyval(np.polyfit(w, f, deg, w=weights), w)
    out = np.full_like(flux, np.nan, dtype=float)
    out[valid] = cont
    return out


# ---------------------------------------------------------------------------
# Method 2 — Penalised Upper-Envelope Spline
# ---------------------------------------------------------------------------

def fit_penalised_upper(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    knot_spacing: float = 8.0,
    n_iter: int = 6,
    keep_fraction_below: float = 0.3,
    smooth_sigma_px: float = 1.5,
    initial_smooth_sigma_px: float = 5.0,
) -> np.ndarray:
    """
    Iterative upper-envelope spline fit.

    Starting from a smooth initial estimate, iteratively discard points that
    fall more than *keep_fraction_below* × scatter below the current fit, then
    refit.  Converges to the upper boundary of the flux distribution.

    Parameters
    ----------
    wave, flux : arrays.
    knot_spacing : interior knot spacing (Angstroms).
    n_iter : iteration count.
    keep_fraction_below : fraction of MAD below the fit to tolerate.
        Smaller values → more aggressive upper-envelope convergence.
    smooth_sigma_px : smoothing after final fit.
    initial_smooth_sigma_px : smoothing for the initial estimate.

    Returns
    -------
    continuum : array of the same shape as *flux*.
    """
    valid = np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(valid) < 10:
        return np.full_like(flux, np.nan, dtype=float)

    w = wave[valid]
    f = flux[valid]
    knots = _knot_positions(w, knot_spacing)

    # Initial estimate: generous smooth.
    cont = _median_smooth(f, initial_smooth_sigma_px)

    for iteration in range(n_iter):
        residual = f - cont
        mad = float(np.nanmedian(np.abs(residual)))
        sigma_est = 1.4826 * mad if mad > 0 else float(np.nanstd(residual))
        if sigma_est <= 0:
            sigma_est = 1.0

        # Keep points near or above the fit.
        threshold = -keep_fraction_below * sigma_est
        keep = residual >= threshold
        n_keep = int(np.count_nonzero(keep))
        if n_keep < max(10, int(0.15 * w.size)):
            # Too aggressive — relax and keep the current fit.
            break

        if knots.size >= 1 and n_keep > knots.size + 4:
            try:
                spl = UnivariateSpline(w[keep], f[keep], t=knots, k=3, ext=0)
                cont = spl(w)
            except Exception:
                coeffs = np.polyfit(w[keep], f[keep], deg=min(3, max(1, n_keep // 8)))
                cont = np.polyval(coeffs, w)
        else:
            coeffs = np.polyfit(w[keep], f[keep], deg=min(3, max(1, n_keep // 8)))
            cont = np.polyval(coeffs, w)

    if smooth_sigma_px > 0:
        cont = _median_smooth(cont, smooth_sigma_px)

    out = np.full_like(flux, np.nan, dtype=float)
    out[valid] = cont
    return out


# ---------------------------------------------------------------------------
# Method 3 — Lorentzian Deblend (Joint Line + Continuum Fit)
# ---------------------------------------------------------------------------

def _lorentzian_cont_model(
    wave: np.ndarray,
    c0: float,
    c1: float,
    c2: float,
    depth: float,
    centre: float,
    gamma: float,
) -> np.ndarray:
    """Quadratic continuum + Lorentzian absorption."""
    cont = c0 + c1 * (wave - centre) + c2 * (wave - centre) ** 2
    lor = depth / (1.0 + ((wave - centre) / gamma) ** 2)
    return cont - lor


def fit_lorentzian_deblend(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    centre_hint: float,
    fwhm_hint: float | None = None,
    ivar: np.ndarray | None = None,
) -> np.ndarray:
    """
    Joint Lorentzian absorption + polynomial continuum fit.

    Fits ``flux = C(lambda) - L(lambda)`` where C is a quadratic polynomial
    and L is a Lorentzian profile.  Returns the **continuum** C(lambda) alone,
    effectively deblending the Balmer absorption from the pseudo-continuum.

    Parameters
    ----------
    wave, flux : arrays.
    centre_hint : approximate Balmer line centre (Angstroms).
    fwhm_hint : approximate FWHM; if None, guessed from data.
    ivar : inverse-variance weights (optional).

    Returns
    -------
    continuum : array of the same shape as *flux*.
    """
    valid = np.isfinite(wave) & np.isfinite(flux)
    if ivar is not None:
        valid &= np.isfinite(ivar) & (ivar > 0)
    if np.count_nonzero(valid) < 8:
        return np.full_like(flux, np.nan, dtype=float)

    w = wave[valid]
    f = flux[valid]

    c0_guess = float(np.nanmedian(f))
    depth_guess = float(max(c0_guess - np.nanmin(f), 1e-3))
    lo, hi = float(w.min()), float(w.max())

    if fwhm_hint is None or not np.isfinite(fwhm_hint):
        fwhm_hint = max(1.0, 0.15 * (hi - lo))
    gamma_guess = max(0.3, float(fwhm_hint) / 2.0)

    p0 = [c0_guess, 0.0, 0.0, depth_guess, float(centre_hint), gamma_guess]
    lower = [0.0, -np.inf, -np.inf, 0.0, lo, 0.1]
    upper = [np.inf, np.inf, np.inf, c0_guess * 3.0, hi, hi - lo]

    sigma = None
    if ivar is not None:
        iv = ivar[valid]
        sigma = np.where(iv > 0, 1.0 / np.sqrt(iv), np.inf)

    try:
        params, _ = curve_fit(
            _lorentzian_cont_model,
            w,
            f,
            p0=p0,
            bounds=(lower, upper),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=15_000,
        )
    except Exception:
        # Fallback: simple high-percentile estimate.
        return np.full_like(flux, float(np.nanpercentile(flux[valid], 90)), dtype=float)

    # Extract the continuum component only (no Lorentzian).
    c0, c1, c2, _depth, ctr, _gamma = params
    cont_all = c0 + c1 * (wave - ctr) + c2 * (wave - ctr) ** 2
    cont_all[~np.isfinite(wave)] = np.nan
    return cont_all


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

BALMER_CENTRES = {
    "HDELTA": 4101.734,
    "HGAMMA": 4340.462,
    "HBETA": 4861.325,
    "HALPHA": 6562.790,
}

BALMER_LINES = ["HDELTA", "HGAMMA", "HBETA", "HALPHA"]
BSUB_BLUE_JUMP_THRESHOLD = 1.0
BSUB_MASK_K_FALLBACKS = (1.0, 0.5, 0.0)


# ---------------------------------------------------------------------------
# Method 4 — Balmer Template Subtraction
# ---------------------------------------------------------------------------

def _fit_absorption_lorentzian(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    centre_hint: float,
    fwhm_hint: float | None = None,
    ivar: np.ndarray | None = None,
    fit_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Fit Lorentzian absorption + linear continuum and return the absorption
    component evaluated at every wavelength in *wave*.

    Returns ``(absorption, meta)`` where *absorption* is non-negative (zero
    where no correction is made) and *meta* contains the fitted parameters.
    """
    zero = np.zeros_like(wave, dtype=float)

    mask = np.isfinite(wave) & np.isfinite(flux)
    if fit_range is not None:
        mask &= (wave >= fit_range[0]) & (wave <= fit_range[1])
    if ivar is not None:
        mask &= np.isfinite(ivar) & (ivar > 0)
    if np.count_nonzero(mask) < 8:
        return zero, {"status": "insufficient_points"}

    w = wave[mask]
    f = flux[mask]
    lo, hi = float(w.min()), float(w.max())

    c0_guess = float(np.nanpercentile(f, 90))
    depth_guess = float(max(c0_guess - np.nanmin(f), 1e-3))
    if fwhm_hint is None or not np.isfinite(fwhm_hint):
        fwhm_hint = max(1.0, 0.15 * (hi - lo))
    gamma_guess = max(0.3, float(fwhm_hint) / 2.0)

    def model(x, c0, c1, depth, ctr, gamma):
        return c0 + c1 * (x - ctr) - depth / (1.0 + ((x - ctr) / gamma) ** 2)

    p0 = [c0_guess, 0.0, depth_guess, float(centre_hint), gamma_guess]
    lower = [0.0, -np.inf, 0.0, lo, 0.1]
    upper = [np.inf, np.inf, c0_guess * 3.0, hi, hi - lo]

    sigma = None
    if ivar is not None:
        iv = ivar[mask]
        sigma = np.where(iv > 0, 1.0 / np.sqrt(iv), np.inf)

    try:
        params, _ = curve_fit(
            model, w, f, p0=p0, bounds=(lower, upper),
            sigma=sigma, absolute_sigma=False, maxfev=15_000,
        )
    except Exception:
        return zero, {"status": "fit_failed"}

    depth, ctr, gamma = params[2], params[3], params[4]

    # Sanity: if depth is negligible, gamma unreasonable, or FWHM too wide,
    # skip.  A FWHM > 40 A is almost certainly degenerate for Balmer lines.
    if depth < 1e-3 or gamma < 0.5:
        return zero, {"status": "degenerate_fit", "depth": depth, "gamma": gamma}
    if 2.0 * gamma > 40.0:
        return zero, {"status": "fwhm_too_wide", "fwhm": 2.0 * gamma}

    # Absorption profile at all wavelengths.
    absorption = depth / (1.0 + ((wave - ctr) / gamma) ** 2)
    absorption[~np.isfinite(wave)] = 0.0
    # Only apply within a generous range around the line centre (5 × gamma)
    # to avoid tiny corrections far from the line biasing the spline.
    far = np.abs(wave - ctr) > 5.0 * gamma
    absorption[far] = 0.0

    meta = {
        "status": "ok",
        "depth": float(depth),
        "centre": float(ctr),
        "gamma": float(gamma),
        "fwhm": float(2.0 * gamma),
    }
    return absorption, meta


def _derivative_jump(
    wave: np.ndarray,
    series: np.ndarray,
    boundary: float,
    *,
    left_width: float = 2.0,
    right_width: float = 2.0,
) -> float:
    """Estimate derivative jump across a boundary."""
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


def _build_balmer_core_mask(
    wave: np.ndarray,
    base_mask: np.ndarray,
    line_meta: dict[str, dict[str, float]],
    *,
    k: float,
) -> np.ndarray:
    """Build a mask excluding +/- 0.5*k*FWHM around Balmer line centres."""
    mask = np.asarray(base_mask, dtype=bool).copy()
    if k <= 0:
        return mask
    for meta in line_meta.values():
        ctr = float(meta["centre"])
        fwhm = float(meta["fwhm"])
        if not np.isfinite(ctr) or not np.isfinite(fwhm) or fwhm <= 0:
            continue
        half = 0.5 * float(k) * fwhm
        lo = ctr - half
        hi = ctr + half
        mask &= ~((wave >= lo) & (wave <= hi))
    return mask


def _blue_boundary_kink(
    wave: np.ndarray,
    continuum: np.ndarray,
    line_meta: dict[str, dict[str, float]],
    *,
    k: float,
) -> float:
    """
    Return the maximum absolute derivative jump at Balmer blue boundaries.
    """
    if k <= 0 or not line_meta:
        return float("nan")
    jumps: list[float] = []
    for meta in line_meta.values():
        ctr = float(meta["centre"])
        fwhm = float(meta["fwhm"])
        if not np.isfinite(ctr) or not np.isfinite(fwhm) or fwhm <= 0:
            continue
        blue = ctr - 0.5 * float(k) * fwhm
        jump = _derivative_jump(wave, continuum, blue)
        if np.isfinite(jump):
            jumps.append(abs(float(jump)))
    if not jumps:
        return float("nan")
    return float(np.nanmax(jumps))


def apply_fitstage_to_balmer_windows(
    fit_mode: str,
    window_results: list[dict[str, object]],
    line_windows: dict[str, tuple[float, float]],
    *,
    # quantile_spline parameters
    qs_tau: float = 0.85,
    qs_knot_spacing: float = 8.0,
    qs_n_iter: int = 15,
    qs_smooth: float = 1.5,
    # penalised_upper parameters
    pu_knot_spacing: float = 8.0,
    pu_n_iter: int = 6,
    pu_keep_frac: float = 0.3,
    pu_smooth: float = 1.5,
    pu_init_smooth: float = 5.0,
    # lorentzian_deblend: no extra params beyond line centres
    # balmer_subtract: needs the sigma-clip continuum fitter
    continuum_fit_func: object | None = None,
    continuum_fit_kwargs: dict | None = None,
    use_fwhm_mask: bool = True,
) -> None:
    """
    Replace the continuum in Balmer windows using the specified fit-stage
    method.  Operates in-place on *window_results*.

    Parameters
    ----------
    fit_mode : one of ``"standard"``, ``"quantile_spline"``,
        ``"penalised_upper"``, ``"lorentzian_deblend"``, ``"balmer_subtract"``.
    window_results : list of per-window dicts produced by
        ``local_normalise_windows()``.
    line_windows : dict mapping line name to (lo, hi) range.
    continuum_fit_func : callable
        The sigma-clip spline fitter (only needed for ``"balmer_subtract"``).
    continuum_fit_kwargs : dict
        Keyword arguments for *continuum_fit_func* (only needed for
        ``"balmer_subtract"``).
    use_fwhm_mask : bool
        If true, ``"balmer_subtract"`` may use Balmer-core FWHM masks during
        refit. If false, the refit always uses the full base mask.
    """
    if fit_mode == "standard":
        return

    for result in window_results:
        group_lines = [str(item) for item in result.get("lines", [])]
        # Only modify groups that contain Balmer lines.
        if not any(line in BALMER_LINES for line in group_lines):
            continue

        wave = np.asarray(result["wave"], dtype=float)
        flux = np.asarray(result["flux"], dtype=float)
        ivar = result.get("ivar")
        if ivar is not None:
            ivar = np.asarray(ivar, dtype=float)

        if wave.size == 0:
            continue

        if fit_mode == "quantile_spline":
            new_cont = fit_quantile_spline(
                wave,
                flux,
                tau=qs_tau,
                knot_spacing=qs_knot_spacing,
                n_iter=qs_n_iter,
                smooth_sigma_px=qs_smooth,
            )

        elif fit_mode == "penalised_upper":
            new_cont = fit_penalised_upper(
                wave,
                flux,
                knot_spacing=pu_knot_spacing,
                n_iter=pu_n_iter,
                keep_fraction_below=pu_keep_frac,
                smooth_sigma_px=pu_smooth,
                initial_smooth_sigma_px=pu_init_smooth,
            )

        elif fit_mode == "lorentzian_deblend":
            # For Lorentzian deblend, fit each Balmer line separately.
            # Build a composite continuum from the per-line fits.
            new_cont = np.asarray(result["continuum"], dtype=float).copy()
            for line in BALMER_LINES:
                if line not in group_lines:
                    continue
                fit_range = line_windows.get(line)
                if fit_range is None:
                    continue
                centre = BALMER_CENTRES.get(line)
                if centre is None:
                    continue

                # Get FWHM hint from existing metadata if available.
                fwhm_hint = None
                fwhm_meta = result.get("fwhm_meta")
                if isinstance(fwhm_meta, dict):
                    lines_meta = fwhm_meta.get("lines")
                    if isinstance(lines_meta, dict) and isinstance(
                        lines_meta.get(line), dict
                    ):
                        fwhm_hint = lines_meta[line].get("fwhm")
                        ctr = lines_meta[line].get("centre")
                        if ctr is not None:
                            centre = float(ctr)

                lr_lo, lr_hi = fit_range
                win_mask = (wave >= lr_lo) & (wave <= lr_hi)
                if np.count_nonzero(win_mask) < 8:
                    continue

                lr_cont = fit_lorentzian_deblend(
                    wave[win_mask],
                    flux[win_mask],
                    centre_hint=float(centre),
                    fwhm_hint=float(fwhm_hint) if fwhm_hint is not None else None,
                    ivar=ivar[win_mask] if ivar is not None else None,
                )
                # Replace only the line-window portion where the fit is valid.
                lr_valid = np.isfinite(lr_cont)
                idx = np.where(win_mask)[0]
                for j, i in enumerate(idx):
                    if lr_valid[j]:
                        new_cont[i] = lr_cont[j]

            # For non-Balmer portions of the group, keep original continuum.
            # (new_cont was initialised from result["continuum"].)

        elif fit_mode == "balmer_subtract":
            # -----------------------------------------------------------
            # Balmer template subtraction.
            #
            # 1. Fit a Lorentzian to each Balmer line → get wing absorption.
            # 2. Add the absorption back to the raw flux → "corrected flux"
            #    that looks like the continuum (+ noise + metal lines) with
            #    the broad Balmer wings removed.
            # 3. Re-run the sigma-clip spline on the corrected flux.
            # 4. Use that spline as the continuum for the *original* flux.
            #
            # This preserves the sigma-clip spline's strength for non-Balmer
            # features while preventing the wing pull-down.
            # -----------------------------------------------------------
            if continuum_fit_func is None:
                raise ValueError(
                    "balmer_subtract requires continuum_fit_func and "
                    "continuum_fit_kwargs"
                )
            cfit_kw = continuum_fit_kwargs or {}

            total_absorption = np.zeros_like(flux, dtype=float)
            for line in BALMER_LINES:
                if line not in group_lines:
                    continue
                fit_range = line_windows.get(line)
                if fit_range is None:
                    continue
                centre = BALMER_CENTRES.get(line)
                if centre is None:
                    continue

                # FWHM hint from preliminary fit metadata.
                fwhm_hint = None
                fwhm_meta = result.get("fwhm_meta")
                if isinstance(fwhm_meta, dict):
                    lines_meta = fwhm_meta.get("lines")
                    if isinstance(lines_meta, dict) and isinstance(
                        lines_meta.get(line), dict
                    ):
                        fwhm_hint = lines_meta[line].get("fwhm")
                        ctr = lines_meta[line].get("centre")
                        if ctr is not None:
                            centre = float(ctr)

                # --- Emission guard ---
                # Check whether the preliminary normalised flux shows
                # emission (flux > continuum) near the line centre.
                # If so, skip subtraction — the Lorentzian cannot model
                # emission and would produce a degenerate fit.
                flux_norm_prelim = np.asarray(
                    result.get("flux_norm", []), dtype=float
                )
                if flux_norm_prelim.size == wave.size:
                    near_centre = (
                        np.abs(wave - centre) < 5.0  # within ~5 A
                    ) & np.isfinite(flux_norm_prelim)
                    if (
                        np.count_nonzero(near_centre) >= 3
                        and float(np.nanmedian(flux_norm_prelim[near_centre]))
                        > 1.05
                    ):
                        # Line is in emission — skip subtraction.
                        continue

                abs_profile, _meta = _fit_absorption_lorentzian(
                    wave,
                    flux,
                    centre_hint=float(centre),
                    fwhm_hint=(
                        float(fwhm_hint) if fwhm_hint is not None else None
                    ),
                    ivar=ivar,
                    fit_range=fit_range,
                )
                total_absorption += abs_profile

            # Wing-corrected flux: add back what the Balmer wings absorbed.
            corrected_flux = flux + total_absorption

            # Build mask for the refit. For problematic stars we may adaptively
            # relax the Balmer-core mask if the blue-edge derivative jump is
            # too large.
            base_mask = np.asarray(result["base_mask"], dtype=bool)
            fwhm_meta = result.get("fwhm_meta")
            line_meta: dict[str, dict[str, float]] = {}
            nominal_k = float("nan")
            if isinstance(fwhm_meta, dict):
                try:
                    nominal_k = float(fwhm_meta.get("k", float("nan")))
                except Exception:
                    nominal_k = float("nan")
                lines_meta = fwhm_meta.get("lines", {})
                if isinstance(lines_meta, dict):
                    for line_name, meta_obj in lines_meta.items():
                        if not isinstance(meta_obj, dict):
                            continue
                        try:
                            ctr = float(meta_obj.get("centre", float("nan")))
                            fwhm = float(meta_obj.get("fwhm", float("nan")))
                        except Exception:
                            continue
                        if np.isfinite(ctr) and np.isfinite(fwhm) and fwhm > 0:
                            line_meta[str(line_name)] = {"centre": ctr, "fwhm": fwhm}

            min_fit_points = int(cfit_kw.get("min_win_px", 21))
            mask_trials: list[dict[str, object]] = []

            if use_fwhm_mask and line_meta and np.isfinite(nominal_k) and nominal_k > 0:
                k_trials: list[float] = [float(nominal_k)]
                for k_alt in BSUB_MASK_K_FALLBACKS:
                    if (k_alt < nominal_k - 1e-6) and (k_alt not in k_trials):
                        k_trials.append(float(k_alt))
                if 0.0 not in k_trials:
                    k_trials.append(0.0)
                for k_trial in k_trials:
                    mask_trials.append(
                        {
                            "mode": "k_trial",
                            "k": float(k_trial),
                            "mask": _build_balmer_core_mask(
                                wave,
                                base_mask,
                                line_meta,
                                k=float(k_trial),
                            ),
                        }
                    )
            else:
                if use_fwhm_mask:
                    fwhm_mask = result.get("fwhm_mask")
                    if isinstance(fwhm_mask, np.ndarray) and fwhm_mask.shape == base_mask.shape:
                        mask_trials.append(
                            {
                                "mode": "provided_mask",
                                "k": float("nan"),
                                "mask": base_mask & fwhm_mask,
                            }
                        )
                mask_trials.append({"mode": "no_mask", "k": 0.0, "mask": base_mask.copy()})

            attempts: list[dict[str, float | int | str]] = []
            successful: list[tuple[float, np.ndarray, dict[str, float | int | str]]] = []
            for trial in mask_trials:
                trial_mask = np.asarray(trial["mask"], dtype=bool)
                n_fit = int(np.count_nonzero(trial_mask & np.isfinite(corrected_flux)))
                trial_info: dict[str, float | int | str] = {
                    "mode": str(trial["mode"]),
                    "k": float(trial["k"]),
                    "n_fit": n_fit,
                }
                if n_fit < max(min_fit_points, 8):
                    trial_info["status"] = "insufficient_points"
                    attempts.append(trial_info)
                    continue

                try:
                    cont_trial, _ = continuum_fit_func(
                        wave,
                        corrected_flux,
                        base_mask=trial_mask,
                        adaptive_smooth=False,
                        **cfit_kw,
                    )
                except Exception:
                    trial_info["status"] = "fit_failed"
                    attempts.append(trial_info)
                    continue

                cont_trial = np.asarray(cont_trial, dtype=float)
                blue_kink = _blue_boundary_kink(
                    wave,
                    cont_trial,
                    line_meta,
                    k=float(trial["k"]),
                )
                trial_info["status"] = "ok"
                trial_info["blue_kink"] = (
                    float(blue_kink) if np.isfinite(blue_kink) else float("nan")
                )
                attempts.append(trial_info)

                if np.isfinite(blue_kink):
                    score = abs(float(blue_kink))
                elif float(trial["k"]) <= 0:
                    score = 0.0
                else:
                    score = float("inf")
                successful.append((score, cont_trial, trial_info))

            chosen_cont: np.ndarray | None = None
            chosen_info: dict[str, float | int | str] | None = None
            for _score, cont_trial, info in successful:
                blue_kink = info.get("blue_kink")
                if (
                    not isinstance(blue_kink, float)
                    or not np.isfinite(blue_kink)
                    or blue_kink <= BSUB_BLUE_JUMP_THRESHOLD
                ):
                    chosen_cont = cont_trial
                    chosen_info = info
                    break

            if chosen_cont is None and successful:
                best = min(successful, key=lambda item: item[0])
                chosen_cont = best[1]
                chosen_info = best[2]

            if chosen_cont is None:
                chosen_cont, _ = continuum_fit_func(
                    wave,
                    corrected_flux,
                    base_mask=base_mask,
                    adaptive_smooth=False,
                    **cfit_kw,
                )
                chosen_cont = np.asarray(chosen_cont, dtype=float)
                chosen_info = {"mode": "fallback_no_mask", "k": 0.0, "status": "ok"}

            if isinstance(result.get("meta"), dict):
                result["meta"]["balmer_subtract_refit"] = {
                    "blue_kink_threshold": float(BSUB_BLUE_JUMP_THRESHOLD),
                    "chosen_mode": str(chosen_info.get("mode", "unknown")),
                    "chosen_k": float(chosen_info.get("k", float("nan"))),
                    "chosen_blue_kink": float(chosen_info.get("blue_kink", float("nan"))),
                    "attempts": attempts,
                }

            new_cont = chosen_cont

        else:
            raise ValueError(f"Unknown fit_mode: {fit_mode!r}")

        # Apply the new continuum.
        safe_cont = np.clip(new_cont, 1e-12, None)
        result["continuum"] = safe_cont
        result["flux_norm"] = flux / safe_cont
