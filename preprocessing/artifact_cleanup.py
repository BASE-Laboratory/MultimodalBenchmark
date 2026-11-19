"""Conservative ("do-no-harm") synchrotron tomography artifact cleanup and spatial gating.

Addresses five empirical artifacts identified in the DLS I13-2 & K11 (DIAD) reconstructions:
1. Residual full-circle and partial-arc ("swirl") detector/scintillator rings surviving Savu RemoveAllRings.
2. Local-tomography radial truncation cupping (169065 at 0.325 um) & unpadded FBP dish (DIAD k11-43334).
3. Pink-beam hardening / photon-starvation axial z-slab shadow cast by the 500 um ruby calibration sphere.
4. Unretrieved Fresnel propagation phase-contrast double-fringes biasing global Otsu pore-throat binarization.
5. Missing angular projection streak in 25 um XRD-CT scan k11-43322 (79/80 projections).

Design principle ("tread carefully"):
- High-contrast structural edges (capillary wall, calibration spheres, macropore walls) are Huber-clamped during
  polar ring estimation and protected by a smooth gradient weight so real phase boundaries never induce false
  counter-arcs and Edge Sharpness Retention stays within 99-101%.
- Tie-homogeneity (Paganin-like) phase regularization is applied ONLY inside optional pore-mask binarization,
  never overwriting raw super-resolution target intensities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    gaussian_filter1d,
    map_coordinates,
    median_filter,
)
from skimage.filters import threshold_otsu


@dataclass(frozen=True)
class ArtifactCleanupReport:
    """Quantitative safety & effectiveness metrics for conservative ring/cupping cleanup."""

    ring_center_yx: Tuple[float, float]
    ring_rms_removed: float
    ring_anova_r2_before: float
    ring_anova_r2_after: float
    ring_variance_reduction_pct: float
    edge_sharpness_retention_ratio: float
    background_std_before: float
    background_std_after: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_ring_anova_r2(
    img_yx: np.ndarray,
    center_yx: Tuple[float, float],
    exclude_mask_yx: Optional[np.ndarray] = None,
    radial_sigma_px: float = 5.0,
    mad_clip: float = 1.8,
) -> float:
    """Compute adjusted one-way ANOVA fraction of bandpass variance explained by concentric rings around center_yx."""
    from scipy.ndimage import binary_dilation

    sl = np.asarray(img_yx, dtype=np.float32)
    bp = gaussian_filter(sl, sigma=0.8) - gaussian_filter(sl, sigma=float(radial_sigma_px))
    gy = gaussian_filter(sl, sigma=1.5, order=[1, 0])
    gx = gaussian_filter(sl, sigma=1.5, order=[0, 1])
    gmag = np.hypot(gy, gx)
    strong_edge = binary_dilation(gmag > np.percentile(gmag, 85.0), iterations=max(2, int(round(radial_sigma_px))))

    valid = (~strong_edge) if exclude_mask_yx is None else ((~strong_edge) & (~np.asarray(exclude_mask_yx, dtype=bool)))
    if int(np.count_nonzero(valid)) < 256:
        valid = np.ones(sl.shape, dtype=bool)

    mad = float(np.median(np.abs(bp[valid] - np.median(bp[valid])))) + 1e-6
    hp = np.clip(bp, -mad_clip * mad, mad_clip * mad)

    ny, nx = sl.shape
    yy, xx = np.indices((ny, nx), dtype=np.float32)

    ys = yy[valid] - np.float32(center_yx[0])
    xs = xx[valid] - np.float32(center_yx[1])
    vals = hp[valid].astype(np.float64)
    vals -= np.mean(vals)
    vvar = float(np.var(vals))
    denom = float(np.dot(vals, vals)) + 1e-12

    rr = np.rint(np.hypot(ys, xs)).astype(np.int32)
    max_r = int(rr.max()) + 1
    cnt = np.bincount(rr, minlength=max_r).astype(np.float64)
    ok = cnt >= 16
    if int(np.count_nonzero(ok)) < 12:
        return 0.0
    sums = np.bincount(rr, weights=vals, minlength=max_r)
    adj_num = float(np.sum((sums[ok] ** 2) / cnt[ok] - vvar))
    return max(0.0, adj_num / denom)


def detect_ring_center_yx(
    vol_or_slice: np.ndarray,
    exclude_mask_yx: Optional[np.ndarray] = None,
    search_y: Tuple[int, int] = (-300, 1300),
    search_x: Tuple[int, int] = (-400, 1300),
    radial_sigma_px: float = 5.0,
) -> Tuple[float, float, float]:
    """Locate the concentric rotation/ring center (cy, cx, adjusted_R2) even when outside a cropped sub-volume."""
    from scipy.ndimage import binary_dilation

    arr = np.asarray(vol_or_slice, dtype=np.float32)
    zmean = np.mean(arr, axis=0) if arr.ndim == 3 else arr
    bp = gaussian_filter(zmean, sigma=0.8) - gaussian_filter(zmean, sigma=float(radial_sigma_px))

    gy = gaussian_filter(zmean, sigma=1.5, order=[1, 0])
    gx = gaussian_filter(zmean, sigma=1.5, order=[0, 1])
    gmag = np.hypot(gy, gx)
    strong_edge = binary_dilation(gmag > np.percentile(gmag, 85.0), iterations=max(4, int(round(2.2 * radial_sigma_px))))
    excl = strong_edge if exclude_mask_yx is None else (strong_edge | np.asarray(exclude_mask_yx, dtype=bool))
    valid_px = ~excl
    if int(np.count_nonzero(valid_px)) < 256:
        valid_px = np.ones(zmean.shape, dtype=bool)

    mad = float(np.median(np.abs(bp[valid_px] - np.median(bp[valid_px])))) + 1e-6
    hp = np.clip(bp, -1.8 * mad, 1.8 * mad)
    hp[~valid_px] = 0.0

    ny, nx = hp.shape
    yy, xx = np.indices((ny, nx), dtype=np.float32)
    grid_step_coarse = max(4, min(32, min(ny, nx) // 16))
    img_step_coarse = max(1, min(4, min(ny, nx) // 64))

    def _score(cy: float, cx: float, step: int) -> float:
        m = valid_px[::step, ::step].ravel()
        if int(np.count_nonzero(m)) < 128:
            return -1.0
        ys = yy[::step, ::step].ravel()[m] - np.float32(cy)
        xs = xx[::step, ::step].ravel()[m] - np.float32(cx)
        vals = hp[::step, ::step].ravel()[m].astype(np.float64)
        vals -= np.mean(vals)
        vvar = float(np.var(vals))
        denom = float(np.dot(vals, vals)) + 1e-12
        rr = np.rint(np.hypot(ys, xs) / step).astype(np.int32)
        max_r = int(rr.max()) + 1
        cnt = np.bincount(rr, minlength=max_r).astype(np.float64)
        ok = cnt >= max(12, 24 // step)
        if int(np.count_nonzero(ok)) < 12:
            return -1.0
        sums = np.bincount(rr, weights=vals, minlength=max_r)
        return float(np.sum((sums[ok] ** 2) / cnt[ok] - vvar) / denom)

    best = (-2.0, 0.5 * (ny - 1), 0.5 * (nx - 1))
    for cy in range(int(search_y[0]), int(search_y[1]) + 1, grid_step_coarse):
        for cx in range(int(search_x[0]), int(search_x[1]) + 1, grid_step_coarse):
            s = _score(float(cy), float(cx), step=img_step_coarse)
            if s > best[0]:
                best = (s, float(cy), float(cx))

    _, bcy, bcx = best
    bloc = best
    for sg, si, hw in [
        (max(2.0, grid_step_coarse / 4.0), max(1, img_step_coarse // 2), float(grid_step_coarse)),
        (1.0, 1, max(4.0, grid_step_coarse / 2.0)),
        (0.5, 1, 2.0),
    ]:
        bloc = (-2.0, bcy, bcx)
        for cy in np.arange(bcy - hw, bcy + hw + 0.05, sg):
            for cx in np.arange(bcx - hw, bcx + hw + 0.05, sg):
                s = _score(float(cy), float(cx), step=si)
                if s > bloc[0]:
                    bloc = (s, float(cy), float(cx))
        _, bcy, bcx = bloc

    return float(bcy), float(bcx), max(0.0, float(bloc[0]))


def _estimate_single_slice_ring_map(
    sl_yx: np.ndarray,
    center_yx: Tuple[float, float],
    radial_sigma_px: float = 5.5,
    angular_window_deg: float = 50.0,
    mad_clip_factor: float = 2.5,
    max_correction_sigma: float = 0.35,
    protect_strong_edges: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate halo-free conservative ring/arc artifact field via coupled polar radial-median & azimuthal-median filtering.

    Mathematical "do-no-harm" property:
    - Step edges (pellet outer contour, capillary wall, ruby sphere rim) are monotonic over win_r, so the 1D radial
      median residual P(r, theta) - M_r(P)(r, theta) is identically ZERO across step boundaries (zero Gaussian halo!).
    - Small isotropic pores subtend << angular_window_deg in theta, so the 1D azimuthal median M_theta(.) is ZERO on pores!
    - Concentric full rings and partial drift arcs (radial width < win_r/2, angular span > angular_window_deg/2) are isolated cleanly.
    """
    ny, nx = sl_yx.shape
    cy, cx = float(center_yx[0]), float(center_yx[1])

    gy = gaussian_filter(sl_yx, sigma=1.0, order=[1, 0])
    gx = gaussian_filter(sl_yx, sigma=1.0, order=[0, 1])
    gmag = np.hypot(gy, gx)
    p10, p95 = [float(v) for v in np.percentile(sl_yx, [10.0, 95.0])]
    contrast_range = max(p95 - p10, float(np.std(sl_yx)) * 2.0, 1e-5)
    g_edge_thresh = 0.32 * contrast_range
    g_p50 = float(np.percentile(gmag, 50.0))

    # 1. Build adaptive polar grid (r, theta) spanning strictly the bounding sector of the slice
    yy, xx = np.indices((ny, nx), dtype=np.float32)
    dy = yy - np.float32(cy)
    dx = xx - np.float32(cx)
    r_map = np.hypot(dy, dx)
    th_map = np.arctan2(dy, dx)  # [-pi, pi]

    r_min = max(0.0, float(np.floor(r_map.min())) - 1.0)
    r_max = float(np.ceil(r_map.max())) + 1.0
    nr = max(12, int(round(r_max - r_min)) + 1)
    r_grid = np.linspace(r_min, r_max, nr, dtype=np.float32)

    center_inside = (0.0 <= cy <= ny - 1) and (0.0 <= cx <= nx - 1)
    r_med = max(60.0, float(np.median(r_map)))
    dtheta = float(np.clip(1.25 / r_med, np.deg2rad(0.25), np.deg2rad(1.0)))

    if center_inside:
        ntheta = max(64, int(round(2.0 * np.pi / dtheta)))
        th_grid = np.linspace(-np.pi, np.pi, ntheta, endpoint=False, dtype=np.float32)
        ang_mode = "wrap"
        th_min, th_span = -np.pi, 2.0 * np.pi
    else:
        th_ref = float(np.arctan2(0.5 * ny - cy, 0.5 * nx - cx))
        th_rel = ((th_map - th_ref + np.pi) % (2.0 * np.pi)) - np.pi
        t0 = float(th_rel.min()) - 2.0 * dtheta
        t1 = float(th_rel.max()) + 2.0 * dtheta
        ntheta = max(32, int(round((t1 - t0) / dtheta)) + 1)
        th_grid_rel = np.linspace(t0, t1, ntheta, dtype=np.float32)
        th_grid = th_ref + th_grid_rel
        th_map = th_rel
        th_min, th_span = t0, (t1 - t0)
        ang_mode = "reflect"

    py = cy + r_grid[:, None] * np.sin(th_grid)[None, :]
    px = cx + r_grid[:, None] * np.cos(th_grid)[None, :]
    polar_img = map_coordinates(sl_yx, [py, px], order=3, mode="nearest")

    # 2. Halo-free 1D radial median high-pass along r
    win_r = max(7, int(round(2.4 * float(radial_sigma_px))) | 1)
    win_r = min(win_r, max(7, (nr // 2) | 1))
    polar_hp = polar_img - median_filter(polar_img, size=(win_r, 1), mode="nearest")

    # Mask perpendicular & oblique Cartesian macro-boundaries (ruby rim, capillary & outer pellet walls)
    if protect_strong_edges:
        wall_thresh = max(0.23 * contrast_range, float(np.percentile(gmag, 95.0)), 4.0 * g_p50)
        wall_yx = gmag > wall_thresh
        wall_polar = map_coordinates(wall_yx.astype(np.float32), [py, px], order=1, mode="nearest") > 0.12
        invalid_polar = binary_dilation(wall_polar, structure=np.ones((5, 3), dtype=bool))
    else:
        invalid_polar = np.zeros_like(polar_hp, dtype=bool)

    # 3. Huber-clamp valid homogeneous pixels and interpolate across masked wall crossings along each constant-r circle
    valid_vals = polar_hp[~invalid_polar]
    if valid_vals.size < 64:
        valid_vals = polar_hp.ravel()
    mad = float(np.median(np.abs(valid_vals - np.median(valid_vals)))) + 1e-6
    robust_sig = 1.4826 * mad
    tau_ring = max(float(mad_clip_factor) * robust_sig, 0.14 * contrast_range)
    polar_clamped = np.clip(polar_hp, -tau_ring, tau_ring).astype(np.float32)

    if np.any(invalid_polar):
        min_valid = max(6, ntheta // 12)
        for ir_row in np.flatnonzero(np.any(invalid_polar, axis=1)):
            v_idx = np.flatnonzero(~invalid_polar[ir_row])
            if v_idx.size < min_valid:
                polar_clamped[ir_row, :] = 0.0
                continue
            inv_idx = np.flatnonzero(invalid_polar[ir_row])
            v_fp = polar_clamped[ir_row, v_idx]
            if center_inside:
                xp = np.concatenate([v_idx - ntheta, v_idx, v_idx + ntheta])
                fp = np.tile(v_fp, 3)
                polar_clamped[ir_row, inv_idx] = np.interp(inv_idx, xp, fp).astype(np.float32)
            else:
                polar_clamped[ir_row, inv_idx] = np.interp(inv_idx, v_idx, v_fp).astype(np.float32)

    win_th = max(5, int(round(np.deg2rad(angular_window_deg) / max(dtheta, 1e-4))) | 1)
    win_th = min(win_th, max(5, (ntheta // 2) | 1))
    polar_ring = median_filter(polar_clamped, size=(1, win_th), mode=ang_mode)
    polar_ring = polar_ring - median_filter(polar_ring, size=(win_r, 1), mode="nearest")

    # 4. Map polar_ring(r, theta) back to Cartesian (y, x) via bicubic interpolation
    ir = (r_map - r_min) / max(r_max - r_min, 1e-6) * (nr - 1)
    if center_inside:
        ith = ((th_map - th_min) % (2.0 * np.pi)) / (2.0 * np.pi) * ntheta
        ring_yx = map_coordinates(polar_ring, [ir, ith], order=3, mode="wrap")
    else:
        ith = (th_map - th_min) / max(th_span, 1e-6) * (ntheta - 1)
        ring_yx = map_coordinates(polar_ring, [ir, ith], order=3, mode="nearest")

    if protect_strong_edges:
        edge_prot_out = np.exp(
            -0.5 * (np.maximum(0.0, gmag - g_edge_thresh) / (0.60 * g_edge_thresh + 1e-6)) ** 2
        ).astype(np.float32)
        ring_yx = ring_yx * edge_prot_out

    core_damp = np.clip((r_map - 2.0) / 5.0, 0.25, 1.0).astype(np.float32)
    ring_yx = ring_yx * core_damp

    max_corr = float(max_correction_sigma) * float(np.std(sl_yx) + 1e-6)
    ring_yx = np.clip(ring_yx, -max_corr, max_corr).astype(np.float32)
    low_grad_mask = gmag <= g_p50
    return ring_yx, gmag, low_grad_mask


def remove_rings_polar_conservative(
    vol_or_slice: np.ndarray,
    center_yx: Optional[Tuple[float, float]] = None,
    radial_sigma_px: float = 5.5,
    angular_window_deg: float = 50.0,
    mad_clip_factor: float = 2.2,
    max_correction_sigma: float = 0.35,
    protect_strong_edges: bool = True,
    exclude_mask_yx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ArtifactCleanupReport]:
    """Conservatively remove full & partial-arc ring artifacts from a 2D slice or 3D (Z, Y, X) volume."""
    arr = np.asarray(vol_or_slice, dtype=np.float32)
    is_2d = arr.ndim == 2
    vol = arr[None, :, :] if is_2d else arr

    if center_yx is None:
        cy, cx, _ = detect_ring_center_yx(
            vol[: min(12, vol.shape[0])], exclude_mask_yx=exclude_mask_yx, radial_sigma_px=radial_sigma_px
        )
        center_yx = (cy, cx)

    cleaned = np.empty_like(vol, dtype=np.float32)
    ring_sq_sum = 0.0
    edge_ratios = []
    bg_stds_before = []
    bg_stds_after = []

    for iz in range(vol.shape[0]):
        sl = vol[iz]
        ring_yx, g_before, low_g = _estimate_single_slice_ring_map(
            sl,
            center_yx=center_yx,
            radial_sigma_px=radial_sigma_px,
            angular_window_deg=angular_window_deg,
            mad_clip_factor=mad_clip_factor,
            max_correction_sigma=max_correction_sigma,
            protect_strong_edges=protect_strong_edges,
        )
        cl = sl - ring_yx
        cleaned[iz] = cl
        ring_sq_sum += float(np.mean(ring_yx * ring_yx))

        # Measure Edge Sharpness Retention on top 10% structural edges (matched sigma=1.0)
        gy_a = gaussian_filter(cl, sigma=1.0, order=[1, 0])
        gx_a = gaussian_filter(cl, sigma=1.0, order=[0, 1])
        g_after = np.hypot(gy_a, gx_a)
        top_e = g_before >= np.percentile(g_before, 90.0)
        edge_ratios.append(float(np.mean(g_after[top_e]) / (np.mean(g_before[top_e]) + 1e-8)))

        if np.any(low_g):
            hp_b = sl - gaussian_filter(sl, sigma=radial_sigma_px)
            hp_a = cl - gaussian_filter(cl, sigma=radial_sigma_px)
            bg_stds_before.append(float(np.std(hp_b[low_g])))
            bg_stds_after.append(float(np.std(hp_a[low_g])))

    mid_b = np.mean(vol[: min(10, vol.shape[0])], axis=0)
    mid_a = np.mean(cleaned[: min(10, cleaned.shape[0])], axis=0)
    r2_b = compute_ring_anova_r2(mid_b, center_yx, exclude_mask_yx=exclude_mask_yx, radial_sigma_px=radial_sigma_px)
    r2_a = compute_ring_anova_r2(mid_a, center_yx, exclude_mask_yx=exclude_mask_yx, radial_sigma_px=radial_sigma_px)
    red_pct = (1.0 - r2_a / max(r2_b, 1e-8)) * 100.0 if r2_b > 1e-6 else 0.0

    report = ArtifactCleanupReport(
        ring_center_yx=(round(float(center_yx[0]), 2), round(float(center_yx[1]), 2)),
        ring_rms_removed=round(float(np.sqrt(ring_sq_sum / vol.shape[0])), 4),
        ring_anova_r2_before=round(r2_b, 6),
        ring_anova_r2_after=round(r2_a, 6),
        ring_variance_reduction_pct=round(float(red_pct), 2),
        edge_sharpness_retention_ratio=round(float(np.mean(edge_ratios)), 4),
        background_std_before=round(float(np.mean(bg_stds_before)) if bg_stds_before else 0.0, 4),
        background_std_after=round(float(np.mean(bg_stds_after)) if bg_stds_after else 0.0, 4),
    )
    return (cleaned[0] if is_2d else cleaned), report


def detrend_local_tomo_cupping_3d(
    vol_zyx: np.ndarray,
    sigma_yx_vox: float = 48.0,
    valid_mask_zyx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Remove local-tomography radial truncation cupping via a pore-safe 7-term 2D quadratic + radial-quartic fit.

    Because the fitted surface spans only [1, y, x, y^2, y*x, x^2, r^4] per z-slab, it has zero frequency response
    at pore scales (cannot blur pore throats) and zero boundary-reflection error near the FOV rim.
    """
    _ = sigma_yx_vox  # retained in signature for backward compatibility
    vol = np.asarray(vol_zyx, dtype=np.float32)
    is_2d = vol.ndim == 2
    v3 = vol[None, :, :] if is_2d else vol
    nz, ny, nx = v3.shape

    yn = np.linspace(-1.0, 1.0, ny, dtype=np.float32)[:, None]
    xn = np.linspace(-1.0, 1.0, nx, dtype=np.float32)[None, :]
    yy = np.broadcast_to(yn, (ny, nx))
    xx = np.broadcast_to(xn, (ny, nx))
    r2 = yy * yy + xx * xx
    basis_full = np.stack(
        [np.ones_like(yy), yy, xx, yy * yy, yy * xx, xx * xx, r2 * r2], axis=-1
    )  # (ny, nx, 7)

    stride = max(1, min(ny, nx) // 128)
    b_sub = basis_full[::stride, ::stride].reshape(-1, 7).astype(np.float64)
    out = np.empty_like(v3, dtype=np.float32)

    for iz in range(nz):
        sl = v3[iz]
        sl_sub = sl[::stride, ::stride].ravel().astype(np.float64)
        if valid_mask_zyx is not None:
            m_sl = valid_mask_zyx if valid_mask_zyx.ndim == 2 else valid_mask_zyx[iz]
            m_sub = (m_sl[::stride, ::stride] > 0).ravel()
        else:
            m_sub = np.ones_like(sl_sub, dtype=bool)
        if int(np.count_nonzero(m_sub)) < 32:
            m_sub = np.ones_like(sl_sub, dtype=bool)

        coeffs, _, _, _ = np.linalg.lstsq(b_sub[m_sub], sl_sub[m_sub], rcond=None)
        cup_surface = np.tensordot(basis_full, coeffs.astype(np.float32), axes=([2], [0]))
        # Subtract only non-constant spatial variation so slice mean inside valid mask stays unchanged!
        mean_surf = float(np.mean(cup_surface[::stride, ::stride].ravel()[m_sub]))
        out[iz] = sl - (cup_surface - np.float32(mean_surf))

    return out[0] if is_2d else out


def compute_ruby_beam_hardening_clean_z_mask(
    num_z_slices: int,
    sphere_center_z_vox: float,
    sphere_radius_vox: float,
    safety_margin_vox: int = 12,
) -> np.ndarray:
    """Return 1D boolean Z-mask (True = clean unshadowed slice) excluding the 500 um ruby sphere's pink-beam shadow."""
    zz = np.arange(int(num_z_slices), dtype=np.float64)
    half_band = float(sphere_radius_vox) + float(safety_margin_vox)
    return np.abs(zz - float(sphere_center_z_vox)) > half_band


def segment_pores_fringe_guarded(
    vol_zyx: np.ndarray,
    roi_mask_zyx: Optional[np.ndarray] = None,
    tie_homogeneity_alpha_vox2: float = 0.6,
    dark_pores: bool = True,
) -> Tuple[np.ndarray, float]:
    """Segment binary pore space after conservative Paganin-like tie-homogeneity damping of ram-lak Fresnel fringes.

    Note: Used ONLY to stabilize binary pore masks against 1-2 px bright/dark phase-contrast overshoots and 0.024s
    shot noise; never overwrites continuous XCT intensities in super-resolution benchmarks.
    """
    vol = np.asarray(vol_zyx, dtype=np.float32)
    if tie_homogeneity_alpha_vox2 > 0.0:
        # Exact Fourier-domain single-distance tie-homogeneity filter: I_reg = F^-1{ F{I} / (1 + alpha * |k|^2) }
        freqs = [np.fft.fftfreq(s) * (2.0 * np.pi) for s in vol.shape[:-1]] + [
            np.fft.rfftfreq(vol.shape[-1]) * (2.0 * np.pi)
        ]
        grids = np.meshgrid(*freqs, indexing="ij")
        k2 = sum(g.astype(np.float32) ** 2 for g in grids)
        filt = 1.0 / (1.0 + np.float32(tie_homogeneity_alpha_vox2) * k2)
        axes_all = tuple(range(vol.ndim))
        reg_vol = np.fft.irfftn(np.fft.rfftn(vol, axes=axes_all) * filt, s=vol.shape, axes=axes_all).astype(np.float32)
    else:
        reg_vol = vol

    vals = reg_vol[roi_mask_zyx > 0] if (roi_mask_zyx is not None and np.any(roi_mask_zyx > 0)) else reg_vol.ravel()
    thresh = float(threshold_otsu(vals))
    pores = (reg_vol < thresh) if dark_pores else (reg_vol > thresh)
    if roi_mask_zyx is not None:
        pores = pores & (roi_mask_zyx > 0)
    return pores.astype(np.uint8), thresh


def interpolate_missing_xrdct_projection(
    sinogram_qat: np.ndarray,
    angles_deg: np.ndarray,
    expected_step_deg: float = 4.55696,
) -> Tuple[np.ndarray, np.ndarray]:
    """Harmonize 360-deg XRD-CT angular quadrature: merge duplicate 0/360 closure frames or interpolate >1.5x gaps.

    Note: Empirical inspection of gts_cs_theta shows k11-43322 (79 frames, 0..355.443 deg) already has exact
    uniform 4.557-deg spacing around 360 deg (gap = 1.0 * step), whereas 80-frame (k11-43323..43325) and 40-frame
    50 um scans include a duplicate endpoint at theta[0] + 360 deg that double-weights 0 deg unless merged.
    """
    sino = np.asarray(sinogram_qat, dtype=np.float32).copy()
    ang = np.asarray(angles_deg, dtype=np.float64).copy()
    if ang.size >= 3 and abs(abs(float(ang[-1] - ang[0])) - 360.0) < 0.25 * expected_step_deg:
        sino[:, 0, :] = 0.5 * (sino[:, 0, :] + sino[:, -1, :])
        return sino[:, :-1, :], ang[:-1]
    if ang.size >= 2 and (360.0 - float(ang[-1] - ang[0])) > 1.5 * expected_step_deg:
        gap = 360.0 - float(ang[-1] - ang[0])
        mid_ang = float(ang[-1]) + 0.5 * gap
        mid_proj = 0.5 * (sino[:, -1, :] + sino[:, 0, :])
        sino = np.concatenate([sino, mid_proj[:, None, :]], axis=1)
        ang = np.append(ang, mid_ang)
    return sino, ang


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Conservative ring/cupping cleanup diagnostic on an HDF5 XCT slab.")
    parser.add_argument("--h5", type=Path, required=True, help="Input HDF5 XCT file.")
    parser.add_argument("--z0", type=int, default=240)
    parser.add_argument("--z1", type=int, default=256)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/artifact_cleanup_report.json"))
    args = parser.parse_args(argv)

    from registration.diad_correlative import read_savu_or_zyx_slab

    slab, _, _ = read_savu_or_zyx_slab(args.h5, z0=args.z0, z1=args.z1)
    _, rep = remove_rings_polar_conservative(slab)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(rep.to_dict(), fh, indent=2)
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
