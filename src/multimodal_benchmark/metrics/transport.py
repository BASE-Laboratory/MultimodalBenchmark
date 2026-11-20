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
    report_threshold_sensitivity: bool = False,
    sensitivity_rel_shifts: tuple[float, ...] = (-0.05, 0.05),
) -> Dict[str, float]:
    """Compare OpenImpala transport properties under one fixed threshold rule applied identically to pred and ref."""
    import hashlib

    vox_m = float(voxel_size_um) * 1e-6
    if inputs_are_grayscale:
        tau_frozen = calibrate_frozen_pore_threshold(ref_vol_or_binary, mask=mask)
        ref_arr = np.asarray(ref_vol_or_binary, dtype=np.float32)
        pred_aligned = affine_cupping_normalize(pred_vol_or_binary, ref_arr, mask=mask)
        ref_bin = (ref_arr < tau_frozen).astype(np.uint8)
        pred_bin = (pred_aligned < tau_frozen).astype(np.uint8)
    else:
        tau_frozen = 0.5
        ref_arr = np.asarray(ref_vol_or_binary, dtype=np.float32)
        pred_aligned = np.asarray(pred_vol_or_binary, dtype=np.float32)
        ref_bin = (ref_arr > 0).astype(np.uint8)
        pred_bin = (pred_aligned > 0).astype(np.uint8)

    def _eval_pair(p_b: np.ndarray, r_b: np.ndarray) -> Dict[str, float]:
        r_key = (r_b.shape, round(vox_m, 12), hashlib.blake2b(r_b.tobytes(), digest_size=8).hexdigest())
        if r_key not in _REF_TRANSPORT_CACHE:
            if len(_REF_TRANSPORT_CACHE) >= 12:
                _REF_TRANSPORT_CACHE.clear()
            _REF_TRANSPORT_CACHE[r_key] = compute_openimpala_transport_metrics(r_b, voxel_size_m=vox_m, pore_phase_id=1)
        mr = _REF_TRANSPORT_CACHE[r_key]
        mp = compute_openimpala_transport_metrics(p_b, voxel_size_m=vox_m, pore_phase_id=1)
        r_deff = abs(mp["d_eff_mean"] - mr["d_eff_mean"]) / max(abs(mr["d_eff_mean"]), 1e-8)
        if np.isfinite(mp["tortuosity_mean"]) and np.isfinite(mr["tortuosity_mean"]):
            r_tau = abs(mp["tortuosity_mean"] - mr["tortuosity_mean"]) / max(abs(mr["tortuosity_mean"]), 1e-8)
        else:
            r_tau = 0.0 if (mp["percolates_any_axis"] == mr["percolates_any_axis"]) else 1.0
        r_ssa = abs(mp["ssa_inv_m"] - mr["ssa_inv_m"]) / max(abs(mr["ssa_inv_m"]), 1e-8)
        return {
            "porosity_pred": mp["porosity"],
            "porosity_ref": mr["porosity"],
            "porosity_abs_err": abs(mp["porosity"] - mr["porosity"]),
            "d_eff_mean_pred": mp["d_eff_mean"],
            "d_eff_mean_ref": mr["d_eff_mean"],
            "d_eff_rel_err": float(r_deff),
            "tortuosity_mean_pred": mp["tortuosity_mean"],
            "tortuosity_mean_ref": mr["tortuosity_mean"],
            "tortuosity_rel_err": float(r_tau),
            "ssa_inv_m_pred": mp["ssa_inv_m"],
            "ssa_inv_m_ref": mr["ssa_inv_m"],
            "ssa_rel_err": float(r_ssa),
            "percolates_all_axes_pred": mp["percolates_all_axes"],
            "percolates_all_axes_ref": mr["percolates_all_axes"],
            "frozen_threshold_value": float(tau_frozen),
        }

    out = _eval_pair(pred_bin, ref_bin)

    if inputs_are_grayscale and report_threshold_sensitivity and len(sensitivity_rel_shifts) > 0:
        lo, hi = np.percentile(ref_arr, [5.0, 95.0])
        dyn = max(float(hi - lo), 1e-6)
        deff_errs = [out["d_eff_rel_err"]]
        eps_errs = [out["porosity_abs_err"]]
        for rel_s in sensitivity_rel_shifts:
            tau_s = tau_frozen + float(rel_s) * dyn
            res_s = _eval_pair((pred_aligned < tau_s).astype(np.uint8), (ref_arr < tau_s).astype(np.uint8))
            deff_errs.append(res_s["d_eff_rel_err"])
            eps_errs.append(res_s["porosity_abs_err"])
        out["threshold_sens_d_eff_rel_err_max"] = float(np.max(deff_errs))
        out["threshold_sens_d_eff_rel_err_span"] = float(np.max(deff_errs) - np.min(deff_errs))
        out["threshold_sens_porosity_abs_err_span"] = float(np.max(eps_errs) - np.min(eps_errs))

    return out
