"""OpenImpala 3D pore-scale transport physics metrics gated behind a frozen HR-calibrated threshold."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from skimage.filters import threshold_otsu

from .image import affine_cupping_normalize


def calibrate_frozen_pore_threshold(
    hr_ref_zyx: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Calibrate the canonical pore/solid threshold strictly on the HR reference volume.

    Applying this frozen threshold (after linear affine dynamic-range normalization) prevents
    super-resolution models from gaming porosity or tortuosity by shifting grayscale contrast.
    """
    r = np.asarray(hr_ref_zyx, dtype=np.float32)
    vals = r[mask > 0] if (mask is not None and np.any(mask > 0)) else r.ravel()
    return float(threshold_otsu(vals))


def binarize_pore_phase_frozen(
    vol_zyx: np.ndarray,
    hr_ref_zyx: np.ndarray,
    frozen_threshold: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    pores_are_dark: bool = True,
) -> np.ndarray:
    """Segment pore space (=1) vs solid (=0) using the frozen HR-calibrated threshold."""
    thresh = frozen_threshold if frozen_threshold is not None else calibrate_frozen_pore_threshold(hr_ref_zyx, mask=mask)
    aligned = affine_cupping_normalize(vol_zyx, hr_ref_zyx, mask=mask)
    pore_mask = (aligned < thresh) if pores_are_dark else (aligned > thresh)
    return pore_mask.astype(np.uint8)


_REF_TRANSPORT_CACHE: Dict[tuple, Dict[str, float]] = {}


def compute_openimpala_transport_metrics(
    binary_pore_zyx: np.ndarray,
    voxel_size_m: float = 1e-6,
    pore_phase_id: int = 1,
) -> Dict[str, float]:
    """Compute 3-axis percolation, porosity, D_eff/D_0, tortuosity, and SSA via OpenImpala."""
    import openimpala as oi

    data = np.asarray(binary_pore_zyx, dtype=np.uint8)
    solid_phase_id = 0 if pore_phase_id == 1 else 1

    with oi.Session():
        vf_res = oi.volume_fraction(data, phase=pore_phase_id)
        eps = float(vf_res.fraction)

        perc = {}
        taus = {}
        deffs = {}
        for axis in ("x", "y", "z"):
            pc = oi.percolation_check(data, phase=pore_phase_id, direction=axis)
            perc[axis] = bool(pc.percolates)
            if pc.percolates:
                try:
                    t_res = oi.tortuosity(data, phase=pore_phase_id, direction=axis, voxel_size=voxel_size_m)
                    taus[axis] = float(t_res.tortuosity)
                    deffs[axis] = float(t_res.effective_diffusivity)
                except Exception:
                    taus[axis] = float("nan")
                    deffs[axis] = 0.0
            else:
                taus[axis] = float("inf")
                deffs[axis] = 0.0

        dxx, dyy, dzz = deffs["x"], deffs["y"], deffs["z"]

        ssa_res = oi.specific_surface_area(
            data,
            phase_a=pore_phase_id,
            phase_b=solid_phase_id,
            voxel_size=voxel_size_m,
        )

    finite_taus = [v for v in taus.values() if np.isfinite(v) and v > 0]
    mean_tau = float(np.mean(finite_taus)) if finite_taus else float("inf")
    mean_deff = float(np.mean([dxx, dyy, dzz]))

    return {
        "porosity": eps,
        "percolates_all_axes": float(all(perc.values())),
        "percolates_any_axis": float(any(perc.values())),
        "d_eff_x": float(dxx),
        "d_eff_y": float(dyy),
        "d_eff_z": float(dzz),
        "d_eff_mean": mean_deff,
        "tortuosity_x": float(taus["x"]),
        "tortuosity_y": float(taus["y"]),
        "tortuosity_z": float(taus["z"]),
        "tortuosity_mean": mean_tau,
        "ssa_inv_m": float(ssa_res.ssa),
    }


def compute_transport_discrepancy(
    pred_vol_or_binary: np.ndarray,
    ref_vol_or_binary: np.ndarray,
    inputs_are_grayscale: bool = True,
    voxel_size_um: float = 0.8125,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compare OpenImpala transport properties between prediction and ground truth."""
    import hashlib

    vox_m = float(voxel_size_um) * 1e-6
    if inputs_are_grayscale:
        tau_frozen = calibrate_frozen_pore_threshold(ref_vol_or_binary, mask=mask)
        ref_bin = (np.asarray(ref_vol_or_binary, dtype=np.float32) < tau_frozen).astype(np.uint8)
        pred_bin = binarize_pore_phase_frozen(pred_vol_or_binary, ref_vol_or_binary, frozen_threshold=tau_frozen, mask=mask)
    else:
        ref_bin = (np.asarray(ref_vol_or_binary) > 0).astype(np.uint8)
        pred_bin = (np.asarray(pred_vol_or_binary) > 0).astype(np.uint8)

    ref_key = (ref_bin.shape, round(vox_m, 12), hashlib.blake2b(ref_bin.tobytes(), digest_size=8).hexdigest())
    if ref_key not in _REF_TRANSPORT_CACHE:
        if len(_REF_TRANSPORT_CACHE) >= 8:
            _REF_TRANSPORT_CACHE.clear()
        _REF_TRANSPORT_CACHE[ref_key] = compute_openimpala_transport_metrics(ref_bin, voxel_size_m=vox_m, pore_phase_id=1)
    m_ref = _REF_TRANSPORT_CACHE[ref_key]
    m_pred = compute_openimpala_transport_metrics(pred_bin, voxel_size_m=vox_m, pore_phase_id=1)

    rel_deff = abs(m_pred["d_eff_mean"] - m_ref["d_eff_mean"]) / max(abs(m_ref["d_eff_mean"]), 1e-8)
    if np.isfinite(m_pred["tortuosity_mean"]) and np.isfinite(m_ref["tortuosity_mean"]):
        rel_tau = abs(m_pred["tortuosity_mean"] - m_ref["tortuosity_mean"]) / max(abs(m_ref["tortuosity_mean"]), 1e-8)
    else:
        rel_tau = 0.0 if (m_pred["percolates_any_axis"] == m_ref["percolates_any_axis"]) else 1.0
    rel_ssa = abs(m_pred["ssa_inv_m"] - m_ref["ssa_inv_m"]) / max(abs(m_ref["ssa_inv_m"]), 1e-8)

    return {
        "porosity_pred": m_pred["porosity"],
        "porosity_ref": m_ref["porosity"],
        "porosity_abs_err": abs(m_pred["porosity"] - m_ref["porosity"]),
        "d_eff_mean_pred": m_pred["d_eff_mean"],
        "d_eff_mean_ref": m_ref["d_eff_mean"],
        "d_eff_rel_err": float(rel_deff),
        "tortuosity_mean_pred": m_pred["tortuosity_mean"],
        "tortuosity_mean_ref": m_ref["tortuosity_mean"],
        "tortuosity_rel_err": float(rel_tau),
        "ssa_rel_err": float(rel_ssa),
        "percolates_all_axes_pred": m_pred["percolates_all_axes"],
        "percolates_all_axes_ref": m_ref["percolates_all_axes"],
    }
