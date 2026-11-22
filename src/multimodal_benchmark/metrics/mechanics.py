"""3D linear elasticity metrics via SMS (finite-volume) and Willot (2015) rotated-stencil FFT-PCG.

Scores whether image super-resolution, de-ringing, or correlative chemical segmentation
alters homogenized effective axial stiffness ``E_eff`` alongside ``OpenImpala`` transport tortuosity,
passing calibrated ``basecoat.Volume`` objects directly through ``sms.VolumeProtocol`` when
``SMS`` is compiled, or falling back automatically to the built-in Willot (2015) rotated-grid
Fourier-Galerkin PCG solver (:mod:`multimodal_benchmark.metrics.fft_elasticity`).
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any, Literal

import numpy as np
from basecoat import Volume

from .fft_elasticity import homogenize_elastic_fft
from .image import affine_cupping_normalize
from .transport import calibrate_frozen_pore_threshold

_REF_MECHANICS_CACHE: dict[tuple[Any, ...], dict[str, float]] = {}

MechanicsBackend = Literal["auto", "sms", "fft"]


def is_sms_available() -> bool:
    """Return True when ``sms`` and its compiled Fortran library ``_libsms_core`` are importable."""
    try:
        from sms import _core as sms_core  # type: ignore[import-not-found]

        sms_core._load_library()
        return True
    except Exception:
        return False


def _resolve_mechanics_backend(backend: MechanicsBackend) -> Literal["sms", "fft"]:
    if backend == "sms":
        return "sms"
    if backend == "fft":
        return "fft"
    if backend != "auto":
        raise ValueError(f"backend must be 'auto', 'sms', or 'fft', got {backend!r}")
    loaded_sms = sys.modules.get("sms")
    if (
        loaded_sms is not None
        and callable(getattr(loaded_sms, "effective_modulus", None))
        and not hasattr(loaded_sms, "_core")
    ):
        return "sms"
    return "sms" if is_sms_available() else "fft"


def compute_fft_mechanics_metrics(
    binary_solid_zyx: np.ndarray | Volume,
    voxel_size_um: float = 0.8125,
    youngs_modulus_pa: float = 10.0e9,
    poisson_ratio: float = 0.25,
    solid_phase_id: int = 1,
    void_contrast: float = 0.0,
    rtol: float = 1e-6,
    max_iter: int = 500,
) -> dict[str, float]:
    """Compute 3-axis homogenized effective axial stiffness ``E_eff`` [Pa] via Willot (2015) rotated-FFT PCG."""
    if isinstance(binary_solid_zyx, Volume):
        vol = (
            binary_solid_zyx
            if binary_solid_zyx.is_calibrated
            else binary_solid_zyx.with_voxel_size_um(float(voxel_size_um))
        )
    else:
        vol = Volume.from_array(
            np.asarray(binary_solid_zyx, dtype=np.uint8), voxel_size_um=float(voxel_size_um)
        )

    res = homogenize_elastic_fft(
        vol,
        youngs_modulus_pa=float(youngs_modulus_pa),
        poisson_ratio=float(poisson_ratio),
        solid_phase=int(solid_phase_id),
        void_contrast=float(void_contrast),
        full_6x6=False,
        rtol=float(rtol),
        max_iter=int(max_iter),
    )
    ex, ey, ez = res.e_eff_xyz_pa
    e_mean = float(np.mean([ex, ey, ez]))
    return {
        "solid_fraction": float(res.solid_fraction),
        "e_eff_x_pa": float(ex),
        "e_eff_y_pa": float(ey),
        "e_eff_z_pa": float(ez),
        "e_eff_mean_pa": e_mean,
        "e_eff_norm_mean": float(e_mean / max(float(youngs_modulus_pa), 1e-12)),
        "sms_converged_all_axes": float(res.converged),
    }


def compute_sms_mechanics_metrics(
    binary_solid_zyx: np.ndarray | Volume,
    voxel_size_um: float = 0.8125,
    youngs_modulus_pa: float = 10.0e9,
    poisson_ratio: float = 0.25,
    solid_phase_id: int = 1,
    rtol: float = 1e-5,
    max_iter: int = 4000,
) -> dict[str, float]:
    """Compute 3-axis homogenized effective axial stiffness ``E_eff`` [Pa] via SMS."""
    import sms  # type: ignore[import-not-found]

    if isinstance(binary_solid_zyx, Volume):
        vol = (
            binary_solid_zyx
            if binary_solid_zyx.is_calibrated
            else binary_solid_zyx.with_voxel_size_um(float(voxel_size_um))
        )
        solid_arr = np.asarray(vol.data)
    else:
        solid_arr = np.asarray(binary_solid_zyx, dtype=np.uint8)
        vol = Volume.from_array(solid_arr, voxel_size_um=float(voxel_size_um))

    solid_frac = float(np.mean(solid_arr == int(solid_phase_id)))
    if solid_frac <= 0.0:
        return {
            "solid_fraction": 0.0,
            "e_eff_x_pa": 0.0,
            "e_eff_y_pa": 0.0,
            "e_eff_z_pa": 0.0,
            "e_eff_mean_pa": 0.0,
            "e_eff_norm_mean": 0.0,
            "sms_converged_all_axes": 1.0,
        }

    e_axes: dict[str, float] = {}
    conv_flags: list[bool] = []
    for axis in ("x", "y", "z"):
        e_val, res = sms.effective_modulus(
            vol,
            direction=axis,
            youngs_modulus=float(youngs_modulus_pa),
            poisson_ratio=float(poisson_ratio),
            solid_phase=int(solid_phase_id),
            rtol=float(rtol),
            max_iter=int(max_iter),
            check_converged=False,
        )
        e_axes[axis] = max(0.0, float(e_val))
        conv_flags.append(bool(res.converged))

    e_mean = float(np.mean([e_axes["x"], e_axes["y"], e_axes["z"]]))
    return {
        "solid_fraction": solid_frac,
        "e_eff_x_pa": e_axes["x"],
        "e_eff_y_pa": e_axes["y"],
        "e_eff_z_pa": e_axes["z"],
        "e_eff_mean_pa": e_mean,
        "e_eff_norm_mean": float(e_mean / max(float(youngs_modulus_pa), 1e-12)),
        "sms_converged_all_axes": float(all(conv_flags)),
    }


def compute_mechanics_metrics(
    binary_solid_zyx: np.ndarray | Volume,
    voxel_size_um: float = 0.8125,
    youngs_modulus_pa: float = 10.0e9,
    poisson_ratio: float = 0.25,
    solid_phase_id: int = 1,
    rtol: float = 1e-5,
    max_iter: int = 4000,
    backend: MechanicsBackend = "auto",
) -> dict[str, float]:
    """Compute 3-axis homogenized ``E_eff`` [Pa] using ``SMS`` or Willot (2015) rotated-FFT PCG."""
    chosen = _resolve_mechanics_backend(backend)
    if chosen == "sms":
        return compute_sms_mechanics_metrics(
            binary_solid_zyx,
            voxel_size_um=voxel_size_um,
            youngs_modulus_pa=youngs_modulus_pa,
            poisson_ratio=poisson_ratio,
            solid_phase_id=solid_phase_id,
            rtol=rtol,
            max_iter=max_iter,
        )
    return compute_fft_mechanics_metrics(
        binary_solid_zyx,
        voxel_size_um=voxel_size_um,
        youngs_modulus_pa=youngs_modulus_pa,
        poisson_ratio=poisson_ratio,
        solid_phase_id=solid_phase_id,
        rtol=min(float(rtol), 1e-5),
        max_iter=min(int(max_iter), 600),
    )


def compute_mechanics_discrepancy(
    pred_vol_or_solid: np.ndarray,
    ref_vol_or_solid: np.ndarray,
    *,
    inputs_are_grayscale: bool = True,
    voxel_size_um: float = 0.8125,
    mask: np.ndarray | None = None,
    youngs_modulus_pa: float = 10.0e9,
    poisson_ratio: float = 0.25,
    pred_solid_lower: np.ndarray | None = None,
    pred_solid_upper: np.ndarray | None = None,
    rtol: float = 1e-5,
    max_iter: int = 4000,
    backend: MechanicsBackend = "auto",
) -> dict[str, float]:
    """Compare homogenized effective stiffness under the frozen HR-calibrated threshold.

    When ``inputs_are_grayscale=True``, applies the exact same HR-calibrated threshold and
    affine cupping normalization as :func:`compute_transport_discrepancy` (with ``solid = ~pore``)
    so ``OpenImpala`` and ``SMS`` / rotated-FFT evaluate identical 3D phase geometries.
    """
    chosen = _resolve_mechanics_backend(backend)
    vox_um = float(voxel_size_um)
    if inputs_are_grayscale:
        tau_frozen = calibrate_frozen_pore_threshold(ref_vol_or_solid, mask=mask)
        ref_arr = np.asarray(ref_vol_or_solid, dtype=np.float32)
        pred_aligned = affine_cupping_normalize(pred_vol_or_solid, ref_arr, mask=mask)
        ref_solid = (ref_arr >= tau_frozen).astype(np.uint8)
        pred_solid = (pred_aligned >= tau_frozen).astype(np.uint8)
    else:
        ref_solid = (np.asarray(ref_vol_or_solid) > 0).astype(np.uint8)
        pred_solid = (np.asarray(pred_vol_or_solid) > 0).astype(np.uint8)

    r_key = (
        chosen,
        ref_solid.shape,
        round(vox_um, 6),
        round(float(youngs_modulus_pa), 2),
        round(float(poisson_ratio), 4),
        hashlib.blake2b(ref_solid.tobytes(), digest_size=8).hexdigest(),
    )
    if r_key not in _REF_MECHANICS_CACHE:
        if len(_REF_MECHANICS_CACHE) >= 12:
            _REF_MECHANICS_CACHE.clear()
        _REF_MECHANICS_CACHE[r_key] = compute_mechanics_metrics(
            ref_solid,
            voxel_size_um=vox_um,
            youngs_modulus_pa=youngs_modulus_pa,
            poisson_ratio=poisson_ratio,
            solid_phase_id=1,
            rtol=rtol,
            max_iter=max_iter,
            backend=chosen,
        )
    mr = _REF_MECHANICS_CACHE[r_key]
    mp = compute_mechanics_metrics(
        pred_solid,
        voxel_size_um=vox_um,
        youngs_modulus_pa=youngs_modulus_pa,
        poisson_ratio=poisson_ratio,
        solid_phase_id=1,
        rtol=rtol,
        max_iter=max_iter,
        backend=chosen,
    )

    rel_e = abs(mp["e_eff_mean_pa"] - mr["e_eff_mean_pa"]) / max(
        abs(mr["e_eff_mean_pa"]), float(youngs_modulus_pa) * 1e-8
    )
    out: dict[str, float] = {
        "solid_fraction_pred": mp["solid_fraction"],
        "solid_fraction_ref": mr["solid_fraction"],
        "solid_fraction_abs_err": abs(mp["solid_fraction"] - mr["solid_fraction"]),
        "e_eff_mean_pred_pa": mp["e_eff_mean_pa"],
        "e_eff_mean_ref_pa": mr["e_eff_mean_pa"],
        "e_eff_norm_pred": mp["e_eff_norm_mean"],
        "e_eff_norm_ref": mr["e_eff_norm_mean"],
        "e_eff_rel_err": float(rel_e),
        "sms_converged_all_axes_pred": mp["sms_converged_all_axes"],
        "sms_converged_all_axes_ref": mr["sms_converged_all_axes"],
    }

    if pred_solid_lower is not None and pred_solid_upper is not None:
        m_lo = compute_mechanics_metrics(
            (np.asarray(pred_solid_lower) > 0).astype(np.uint8),
            voxel_size_um=vox_um,
            youngs_modulus_pa=youngs_modulus_pa,
            poisson_ratio=poisson_ratio,
            solid_phase_id=1,
            rtol=rtol,
            max_iter=max_iter,
            backend=chosen,
        )
        m_hi = compute_mechanics_metrics(
            (np.asarray(pred_solid_upper) > 0).astype(np.uint8),
            voxel_size_um=vox_um,
            youngs_modulus_pa=youngs_modulus_pa,
            poisson_ratio=poisson_ratio,
            solid_phase_id=1,
            rtol=rtol,
            max_iter=max_iter,
            backend=chosen,
        )
        lo_pa = min(m_lo["e_eff_mean_pa"], m_hi["e_eff_mean_pa"])
        hi_pa = max(m_lo["e_eff_mean_pa"], m_hi["e_eff_mean_pa"])
        out["e_eff_conformal_lower_pa"] = float(lo_pa)
        out["e_eff_conformal_upper_pa"] = float(hi_pa)
        out["e_eff_conformal_interval_pa"] = float(hi_pa - lo_pa)
        out["e_eff_conformal_covers_ref"] = float(lo_pa <= mr["e_eff_mean_pa"] <= hi_pa)

    return out
