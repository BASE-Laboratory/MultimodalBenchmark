"""Conservative ("do-no-harm") synchrotron tomography artifact cleanup and spatial gating.

Generic polar ring removal, ring-center detection, and 2D radial cupping detrending delegate
to ``basecoat`` (``v0.3.0+``), while dataset/beamline-specific helpers (500 um ruby pink-beam
shadow z-masking, Paganin-like Fresnel fringe binarization, and 25 um XRD-CT angular projection
closure) remain in ``MultimodalBenchmark`` (Layer 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from basecoat import (
    ArtifactCleanupReport,
    compute_ring_anova_r2 as _bc_compute_ring_anova_r2,
    detect_ring_center_yx as _bc_detect_ring_center_yx,
    detrend_radial_cupping as _bc_detrend_radial_cupping,
    remove_rings_polar_conservative as _bc_remove_rings_polar_conservative,
)
from skimage.filters import threshold_otsu


def compute_ring_anova_r2(
    img_yx: np.ndarray,
    center_yx: Tuple[float, float],
    exclude_mask_yx: Optional[np.ndarray] = None,
    radial_sigma_px: float = 5.0,
    mad_clip: float = 1.8,
) -> float:
    """Compute adjusted one-way ANOVA fraction of bandpass variance explained by concentric rings around center_yx."""
    return _bc_compute_ring_anova_r2(
        img_yx,
        center_yx,
        exclude_mask_yx=exclude_mask_yx,
        radial_sigma_px=radial_sigma_px,
        mad_clip=mad_clip,
    )


def detect_ring_center_yx(
    vol_or_slice: np.ndarray,
    exclude_mask_yx: Optional[np.ndarray] = None,
    search_y: Tuple[int, int] = (-300, 1300),
    search_x: Tuple[int, int] = (-400, 1300),
    radial_sigma_px: float = 5.0,
) -> Tuple[float, float, float]:
    """Locate the concentric rotation/ring center (cy, cx, adjusted_R2) even when outside a cropped sub-volume."""
    return _bc_detect_ring_center_yx(
        vol_or_slice,
        exclude_mask_yx=exclude_mask_yx,
        search_y=search_y,
        search_x=search_x,
        radial_sigma_px=radial_sigma_px,
    )


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
    """Conservatively remove full and partial-arc ring artifacts via ``basecoat.remove_rings_polar_conservative``."""
    arr = np.asarray(vol_or_slice, dtype=np.float32)
    if center_yx is None:
        v_sub = arr[None, :, :] if arr.ndim == 2 else arr[: min(12, arr.shape[0])]
        cy, cx, _ = detect_ring_center_yx(
            v_sub,
            exclude_mask_yx=exclude_mask_yx,
            radial_sigma_px=radial_sigma_px,
        )
        center_yx = (cy, cx)
    return _bc_remove_rings_polar_conservative(
        arr,
        center_yx=center_yx,
        radial_sigma_px=radial_sigma_px,
        angular_window_deg=angular_window_deg,
        mad_clip_factor=mad_clip_factor,
        max_correction_sigma=max_correction_sigma,
        protect_strong_edges=protect_strong_edges,
        exclude_mask_yx=exclude_mask_yx,
    )


def detrend_local_tomo_cupping_3d(
    vol_zyx: np.ndarray,
    sigma_yx_vox: float = 48.0,
    valid_mask_zyx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Remove low-order 2D quadratic + radial-quartic [1, y, x, y^2, y*x, x^2, r^4] background curvature.

    WARNING (Chemistry Confounding — Frozen Scope Policy):
    Empirical comparison against 25 um XRD-CT shows that in core-shell ion-exchanged pellets (Zn-rich outer shell,
    Na-rich core), a 2D radial polynomial fit correlates strongly with real Zn/Na photoelectric attenuation
    (r = +0.28..+0.59, inverting radial chemistry slopes). Do NOT apply this function to official benchmark
    reference volumes (`raw_float32` or `cleaned_dering_v1`); restrict strictly to diagnostic visualization on
    homogeneous samples.
    """
    return _bc_detrend_radial_cupping(
        np.asarray(vol_zyx, dtype=np.float32),
        sigma_yx_vox=sigma_yx_vox,
        valid_mask_zyx=valid_mask_zyx,
    )


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
    """Segment binary pore space after conservative Paganin-like tie-homogeneity damping of ram-lak Fresnel fringes."""
    vol = np.asarray(vol_zyx, dtype=np.float32)
    if tie_homogeneity_alpha_vox2 > 0.0:
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
    """Harmonize 360-deg XRD-CT angular quadrature: merge duplicate 0/360 closure frames or interpolate >1.5x gaps."""
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
