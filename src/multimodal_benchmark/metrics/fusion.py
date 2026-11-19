"""T3 correlative chemical fusion metrics: noise-normalized chi^2_nu, 25 um accuracy, and HR physical consistency."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import downscale_local_mean


def compute_noise_normalized_chi2(
    pred_map: np.ndarray,
    ref_map: np.ndarray,
    ref_sigma2: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute reduced chi-square chi^2_nu against the 25 um XRD-CT reference noise floor sigma^2_ref(x,y).

    chi^2_nu ~ 1.0 indicates agreement down to the empirical synchrotron diffraction noise floor;
    values < 1.0 warn against overfitting reconstruction streak noise.
    """
    p = np.asarray(pred_map, dtype=np.float64)
    r = np.asarray(ref_map, dtype=np.float64)
    s2 = np.maximum(np.asarray(ref_sigma2, dtype=np.float64), 1e-8)
    valid = (mask > 0) if mask is not None else np.ones(r.shape, dtype=bool)
    ratio = ((p[valid] - r[valid]) ** 2) / s2[valid]
    return float(np.mean(ratio))


def _channel_25um_metrics(
    pred_2d: np.ndarray,
    ref_2d: np.ndarray,
    sigma2_2d: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, float]:
    p = np.asarray(pred_2d, dtype=np.float64)
    r = np.asarray(ref_2d, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    pearson_r = float(np.corrcoef(p.ravel(), r.ravel())[0, 1]) if (np.std(p) > 1e-12 and np.std(r) > 1e-12) else 0.0
    dyn = max(float(np.ptp(r)), 1e-6)
    win = min(7, min(r.shape) if min(r.shape) % 2 == 1 else min(r.shape) - 1)
    ssim_val = float(structural_similarity(r, p, data_range=dyn, win_size=max(3, win)))

    out = {
        f"{prefix}rmse": rmse,
        f"{prefix}pearson_r": pearson_r,
        f"{prefix}ssim": ssim_val,
    }
    if sigma2_2d is not None:
        out[f"{prefix}reduced_chi2"] = compute_noise_normalized_chi2(p, r, sigma2_2d)
    return out


def compute_t3_fusion_metrics(
    pred_zn_25um: np.ndarray,
    pred_na_25um: np.ndarray,
    ref_zn_25um: np.ndarray,
    ref_na_25um: np.ndarray,
    ref_zn_sigma2: Optional[np.ndarray] = None,
    ref_na_sigma2: Optional[np.ndarray] = None,
    hr_pred_zn: Optional[np.ndarray] = None,
    hr_pred_na: Optional[np.ndarray] = None,
    hr_pore_mask: Optional[np.ndarray] = None,
    hr_solid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate T3a (25 um ground-truth chemical maps) and optional T3b (HR physical consistency)."""
    zn_p = np.asarray(pred_zn_25um, dtype=np.float64)
    na_p = np.asarray(pred_na_25um, dtype=np.float64)
    zn_r = np.asarray(ref_zn_25um, dtype=np.float64)
    na_r = np.asarray(ref_na_25um, dtype=np.float64)

    if zn_p.ndim == 2:
        zn_p = zn_p[None, ...]
        na_p = na_p[None, ...]
        zn_r = zn_r[None, ...]
        na_r = na_r[None, ...]
        if ref_zn_sigma2 is not None and ref_zn_sigma2.ndim == 2:
            ref_zn_sigma2 = ref_zn_sigma2[None, ...]
        if ref_na_sigma2 is not None and ref_na_sigma2.ndim == 2:
            ref_na_sigma2 = ref_na_sigma2[None, ...]

    slice_metrics = []
    for iz in range(zn_r.shape[0]):
        sz = ref_zn_sigma2[iz] if ref_zn_sigma2 is not None else None
        sn = ref_na_sigma2[iz] if ref_na_sigma2 is not None else None
        m_zn = _channel_25um_metrics(zn_p[iz], zn_r[iz], sz, prefix="zn_")
        m_na = _channel_25um_metrics(na_p[iz], na_r[iz], sn, prefix="na_")
        slice_metrics.append({**m_zn, **m_na})

    keys = list(slice_metrics[0].keys())
    out: Dict[str, float] = {k: float(np.mean([sm[k] for sm in slice_metrics])) for k in keys}
    out["mean_rmse"] = 0.5 * (out["zn_rmse"] + out["na_rmse"])
    out["mean_pearson_r"] = 0.5 * (out["zn_pearson_r"] + out["na_pearson_r"])
    out["mean_ssim"] = 0.5 * (out["zn_ssim"] + out["na_ssim"])
    if "zn_reduced_chi2" in out and "na_reduced_chi2" in out:
        out["mean_reduced_chi2"] = 0.5 * (out["zn_reduced_chi2"] + out["na_reduced_chi2"])

    # Optional T3b HR physical consistency diagnostics
    if hr_pred_zn is not None and hr_pred_na is not None:
        hz = np.asarray(hr_pred_zn, dtype=np.float64)
        hn = np.asarray(hr_pred_na, dtype=np.float64)
        if hr_pore_mask is not None and np.any(hr_pore_mask > 0):
            pm = hr_pore_mask > 0
            solid_scale = float(np.percentile(np.abs(hz) + np.abs(hn), 90.0)) + 1e-8
            out["t3b_pore_leakage_rel"] = float(np.mean(np.abs(hz[pm]) + np.abs(hn[pm])) / solid_scale)
        if hr_solid_mask is not None and np.any(hr_solid_mask > 0):
            sm = hr_solid_mask > 0
            tot_solid = hz[sm] + hn[sm]
            out["t3b_solid_stoichiometry_cv"] = float(np.std(tot_solid) / (np.abs(np.mean(tot_solid)) + 1e-8))
        # Forward-PSF conservation check if HR grid is an integer multiple of 25 um slice grid
        if hz.ndim == 2 and zn_r.shape[0] == 1:
            fy = hz.shape[0] // zn_r.shape[1]
            fx = hz.shape[1] // zn_r.shape[2]
            if fy >= 1 and fx >= 1 and fy * zn_r.shape[1] == hz.shape[0] and fx * zn_r.shape[2] == hz.shape[1]:
                fwd_zn = downscale_local_mean(hz, (fy, fx))
                fwd_na = downscale_local_mean(hn, (fy, fx))
                out["t3b_forward_psf_rmse"] = 0.5 * (
                    float(np.sqrt(np.mean((fwd_zn - zn_r[0]) ** 2)))
                    + float(np.sqrt(np.mean((fwd_na - na_r[0]) ** 2)))
                )

    return out
