"""Frozen benchmark 7-DOF transform loader & synthetic-vs-real pair generator for BASE-Laboratory/OpenLSR-X.

Per the Zn-13X-Bench protocol:
1. OpenLSR-X data loaders MUST consume the frozen benchmark 7-DOF similarity transforms from
   `configs/i13_benchmark_transforms.json` (`load_benchmark_7dof_transform` / `resample_with_benchmark_transform`)
   rather than running per-experiment or integer-ratio registration.
2. For the Synthetic-vs-Real Headline Experiment (`T1`), `synthesize_degraded_lr_from_hr` generates matched
   synthetic low-resolution inputs from the high-resolution rung at the exact true magnification ratio
   (1.604x, 2.026x, 2.493x, 5.051x, 8.101x), with and without calibrated Poisson-Gaussian detector noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import SimpleITK as sitk

from registration.i13_ladder import (
    LadderRegistrationResult,
    SimilarityTransform3D,
    radial_non_cupping_mask,
    register_3d_similarity,
    resample_moving_to_fixed_zyx,
)

DEFAULT_BENCHMARK_TRANSFORMS_JSON = Path(__file__).resolve().parents[2] / "configs" / "i13_benchmark_transforms.json"

_TRACK_ALIASES = {
    "1.6x": "1p604x",
    "1p6x": "1p604x",
    "1p604x": "1p604x",
    "2x": "2p026x",
    "2.026x": "2p026x",
    "2p026x": "2p026x",
    "2.5x": "2p493x",
    "2p5x": "2p493x",
    "2p493x": "2p493x",
    "5x": "5p051x",
    "5.051x": "5p051x",
    "5p051x": "5p051x",
    "8x": "8p101x",
    "8.101x": "8p101x",
    "8p101x": "8p101x",
}


def load_benchmark_7dof_transform(
    track: str = "2p026x",
    transforms_json: Path | str = DEFAULT_BENCHMARK_TRANSFORMS_JSON,
    center_xyz_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    include_zenodo_8bit_crop_offset: bool = False,
) -> Tuple[SimilarityTransform3D, Dict[str, Any]]:
    """Load the frozen Zn-13X-Bench 7-DOF similarity transform for an I13-2 ladder track."""
    key = _TRACK_ALIASES.get(track.strip().lower(), track.strip())
    data = json.loads(Path(transforms_json).read_text(encoding="utf-8"))
    if key not in data["tracks"]:
        raise KeyError(f"Unknown T1 track '{track}'. Available tracks: {sorted(data['tracks'].keys())}")
    meta = dict(data["tracks"][key])

    euler = meta["euler_xyz_rad"]
    s3d = sitk.Similarity3DTransform()
    s3d.SetCenter([float(c) for c in center_xyz_um])
    # Convert small-angle Euler (rx, ry, rz) to unit versor via SimpleITK Euler3DTransform matrix
    e3d = sitk.Euler3DTransform()
    e3d.SetRotation(float(euler[0]), float(euler[1]), float(euler[2]))
    s3d.SetMatrix(e3d.GetMatrix())
    s3d.SetScale(float(meta["fitted_scale_factor_lr_over_hr_nom"]))

    tx, ty, tz = [float(v) for v in meta["fine_translation_xyz_um"]]
    if include_zenodo_8bit_crop_offset:
        cz, cy, cx = [float(v) for v in meta["zenodo_8bit_crop_offset_zyx_um"]]
        tx += cx
        ty += cy
        tz += cz
    s3d.SetTranslation([tx, ty, tz])
    tfm = SimilarityTransform3D.from_sitk(s3d)
    return tfm, meta


def resample_with_benchmark_transform(
    lr_zyx: np.ndarray,
    hr_shape_zyx: Tuple[int, int, int],
    lr_nominal_voxel_um: float,
    hr_nominal_voxel_um: float,
    track: str = "2p026x",
    transforms_json: Path | str = DEFAULT_BENCHMARK_TRANSFORMS_JSON,
    include_zenodo_8bit_crop_offset: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resample LR scan onto HR grid using ONLY the frozen Zn-13X-Bench 7-DOF transform (no re-registration)."""
    cx = 0.5 * (hr_shape_zyx[2] - 1) * float(hr_nominal_voxel_um)
    cy = 0.5 * (hr_shape_zyx[1] - 1) * float(hr_nominal_voxel_um)
    cz = 0.5 * (hr_shape_zyx[0] - 1) * float(hr_nominal_voxel_um)
    tfm, meta = load_benchmark_7dof_transform(
        track=track,
        transforms_json=transforms_json,
        center_xyz_um=(cx, cy, cz),
        include_zenodo_8bit_crop_offset=include_zenodo_8bit_crop_offset,
    )
    lr_on_hr = resample_moving_to_fixed_zyx(
        moving_zyx=np.asarray(lr_zyx, dtype=np.float32),
        moving_spacing_zyx_um=(float(lr_nominal_voxel_um),) * 3,
        fixed_shape_zyx=hr_shape_zyx,
        fixed_spacing_zyx_um=(float(hr_nominal_voxel_um),) * 3,
        transform=tfm,
    )
    return lr_on_hr, meta


def synthesize_degraded_lr_from_hr(
    hr_zyx: np.ndarray,
    true_scale_ratio: float,
    with_noise: bool = False,
    noise_sigma_rel: float = 0.035,
    psf_sigma_factor: float = 0.45,
    seed: int = 42,
    return_on_hr_grid: bool = False,
) -> np.ndarray:
    """Synthesize degraded LR input from HR reference at `true_scale_ratio` for the Synthetic-vs-Real experiment."""
    hr = np.asarray(hr_zyx, dtype=np.float32)
    sigma_hr = float(psf_sigma_factor) * float(true_scale_ratio)
    blurred = gaussian_filter(hr, sigma=sigma_hr, mode="nearest")
    inv_s = 1.0 / float(true_scale_ratio)
    lr_syn = zoom(blurred, zoom=(inv_s, inv_s, inv_s), order=1, mode="nearest", grid_mode=True).astype(np.float32)

    if with_noise:
        rng = np.random.default_rng(seed)
        p5, p95 = [float(v) for v in np.percentile(lr_syn, [5.0, 95.0])]
        dyn = max(p95 - p5, float(np.std(lr_syn)) * 2.0, 1e-6)
        # Mixed Poisson-like signal-dependent + Gaussian readout noise matching synchrotron sCMOS statistics
        rel_intensity = np.clip((lr_syn - p5) / dyn, 0.05, 2.0)
        sig_map = float(noise_sigma_rel) * dyn * np.sqrt(0.4 + 0.6 * rel_intensity)
        lr_syn = (lr_syn + rng.normal(0.0, 1.0, size=lr_syn.shape).astype(np.float32) * sig_map).astype(np.float32)

    if return_on_hr_grid:
        factors = tuple(float(t) / float(s) for t, s in zip(hr.shape, lr_syn.shape))
        return zoom(lr_syn, zoom=factors, order=3, mode="nearest", grid_mode=True).astype(np.float32)
    return lr_syn


def align_openlsrx_7dof_pair(
    lr_zyx: np.ndarray,
    hr_zyx: np.ndarray,
    lr_voxel_um: float,
    hr_voxel_um: float,
    apply_local_tomo_mask: bool = False,
    track: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Align an LR/HR pair using frozen benchmark transform when `track` is given, else fit 7-DOF NMI."""
    hr = np.asarray(hr_zyx, dtype=np.float32)
    lr = np.asarray(lr_zyx, dtype=np.float32)
    if track is not None:
        lr_on_hr, meta = resample_with_benchmark_transform(
            lr, hr.shape, lr_voxel_um, hr_voxel_um, track=track
        )
        return lr_on_hr, hr, meta

    hr_sp = (float(hr_voxel_um),) * 3
    lr_sp = (float(lr_voxel_um),) * 3
    mask = radial_non_cupping_mask(hr.shape, valid_radius_fraction=0.82) if apply_local_tomo_mask else None
    reg_res: LadderRegistrationResult = register_3d_similarity(
        fixed_zyx=hr,
        moving_zyx=lr,
        fixed_spacing_zyx_um=hr_sp,
        moving_spacing_zyx_um=lr_sp,
        fixed_mask_zyx=mask,
        metric="mattes_mi",
        allow_scale=True,
    )
    lr_resampled_on_hr = resample_moving_to_fixed_zyx(
        moving_zyx=lr,
        moving_spacing_zyx_um=lr_sp,
        fixed_shape_zyx=hr.shape,
        fixed_spacing_zyx_um=hr_sp,
        transform=reg_res.transform,
    )
    return lr_resampled_on_hr, hr, reg_res.to_dict()
