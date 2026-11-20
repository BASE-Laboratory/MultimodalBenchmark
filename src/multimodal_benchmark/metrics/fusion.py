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


def lowpass_2d_to_resolution_cutoff(
    img_2d: np.ndarray,
    cutoff_cyc_per_px: float = 0.43,
    rolloff_cyc_per_px: float = 0.05,
) -> np.ndarray:
    """Band-limit a 2D map to the measured XRD-CT reference resolution cutoff via a raised-cosine Fourier filter."""
    arr = np.asarray(img_2d, dtype=np.float64)
    ny, nx = arr.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    kr = np.hypot(fy, fx)

    k0 = max(0.05, float(cutoff_cyc_per_px) - 0.5 * float(rolloff_cyc_per_px))
    k1 = float(cutoff_cyc_per_px) + 0.5 * float(rolloff_cyc_per_px)
    filt = np.ones_like(kr, dtype=np.float64)
    trans = (kr > k0) & (kr < k1)
    filt[trans] = 0.5 * (1.0 + np.cos(np.pi * (kr[trans] - k0) / max(k1 - k0, 1e-6)))
    filt[kr >= k1] = 0.0
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * filt)).astype(np.float64)


def _channel_25um_metrics(
    pred_2d: np.ndarray,
    ref_2d: np.ndarray,
    sigma2_2d: Optional[np.ndarray] = None,
    prefix: str = "",
    cutoff_cyc_per_px: float = 0.43,
) -> Dict[str, float]:
    p = np.asarray(pred_2d, dtype=np.float64)
    r = np.asarray(ref_2d, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    pearson_r = float(np.corrcoef(p.ravel(), r.ravel())[0, 1]) if (np.std(p) > 1e-12 and np.std(r) > 1e-12) else 0.0
    dyn = max(float(np.ptp(r)), 1e-6)
    win = min(7, min(r.shape) if min(r.shape) % 2 == 1 else min(r.shape) - 1)
    ssim_val = float(structural_similarity(r, p, data_range=dyn, win_size=max(3, win)))

    # Band-limited evaluation up to measured 25 um XRD-CT resolution limit (~58 um at SB-FRC 0.5 / 1/2-bit -> 0.43 cyc/px)
    p_bl = lowpass_2d_to_resolution_cutoff(p, cutoff_cyc_per_px=cutoff_cyc_per_px)
    r_bl = lowpass_2d_to_resolution_cutoff(r, cutoff_cyc_per_px=cutoff_cyc_per_px)
    bl_rmse = float(np.sqrt(np.mean((p_bl - r_bl) ** 2)))
    bl_pearson = (
        float(np.corrcoef(p_bl.ravel(), r_bl.ravel())[0, 1])
        if (np.std(p_bl) > 1e-12 and np.std(r_bl) > 1e-12)
        else 0.0
    )

    out = {
        f"{prefix}rmse": rmse,
        f"{prefix}pearson_r": pearson_r,
        f"{prefix}ssim": ssim_val,
        f"{prefix}bandlimited_rmse": bl_rmse,
        f"{prefix}bandlimited_pearson_r": bl_pearson,
    }
    if sigma2_2d is not None:
        out[f"{prefix}reduced_chi2"] = compute_noise_normalized_chi2(p, r, sigma2_2d)
        # Scale pixelwise variance by retained Fourier power fraction pi * k_cut^2 / 1.0 (~0.58)
        bw_frac = min(1.0, float(np.pi * (cutoff_cyc_per_px**2)))
        out[f"{prefix}bandlimited_reduced_chi2"] = compute_noise_normalized_chi2(
            p_bl, r_bl, np.asarray(sigma2_2d, dtype=np.float64) * max(bw_frac, 0.1)
        )
    return out


def _slice_reaction_front_and_phase_corr(
    zn_p_2d: np.ndarray,
    na_p_2d: np.ndarray,
    zn_r_2d: np.ndarray,
    na_r_2d: np.ndarray,
    pixel_size_um: float = 25.0,
) -> Dict[str, float]:
    """Compute in-pellet Na/Zn correlation and Zn/(Zn+Na) reaction-front penetration position error (in um)."""
    from scipy.ndimage import distance_transform_edt

    zp = np.clip(np.asarray(zn_p_2d, dtype=np.float64), 0.0, None)
    np_ = np.clip(np.asarray(na_p_2d, dtype=np.float64), 0.0, None)
    zr = np.clip(np.asarray(zn_r_2d, dtype=np.float64), 0.0, None)
    nr = np.clip(np.asarray(na_r_2d, dtype=np.float64), 0.0, None)

    tot_r = zr + nr
    p95 = float(np.percentile(tot_r, 95.0)) if tot_r.size > 0 else 0.0
    pellet = tot_r > max(0.20 * p95, 1e-8)
    if int(np.count_nonzero(pellet)) < 8:
        pellet = np.ones_like(tot_r, dtype=bool)

    corr_ref = (
        float(np.corrcoef(zr[pellet], nr[pellet])[0, 1])
        if (np.std(zr[pellet]) > 1e-10 and np.std(nr[pellet]) > 1e-10)
        else 0.0
    )
    corr_pred = (
        float(np.corrcoef(zp[pellet], np_[pellet])[0, 1])
        if (np.std(zp[pellet]) > 1e-10 and np.std(np_[pellet]) > 1e-10)
        else 0.0
    )

    # Distance inward from outer catalyst pellet boundary (in um)
    depth_um = distance_transform_edt(pellet) * float(pixel_size_um)
    f_zn_r = zr / np.maximum(zr + nr, 1e-8)
    f_zn_p = zp / np.maximum(zp + np_, 1e-8)

    # If slice straddles a reaction front (both >0.5 and <=0.5 present inside pellet), measure 0.5-isocontour distance;
    # otherwise fall back to Zn-weighted mean penetration depth in um so pure-Zn slices remain well-defined.
    has_front_r = bool(np.any(f_zn_r[pellet] >= 0.5) and np.any(f_zn_r[pellet] < 0.5))
    has_front_p = bool(np.any(f_zn_p[pellet] >= 0.5) and np.any(f_zn_p[pellet] < 0.5))
    if has_front_r and has_front_p:
        # Mean symmetric closest-point distance (Chamfer / Hausdorff-mean in um) between f_zn == 0.5 boundaries
        b_r = pellet & (np.abs(f_zn_r - 0.5) <= 0.18)
        b_p = pellet & (np.abs(f_zn_p - 0.5) <= 0.18)
        if np.any(b_r) and np.any(b_p):
            dt_to_r = distance_transform_edt(~b_r) * float(pixel_size_um)
            dt_to_p = distance_transform_edt(~b_p) * float(pixel_size_um)
            front_err_um = 0.5 * (float(np.mean(dt_to_r[b_p])) + float(np.mean(dt_to_p[b_r])))
        else:
            w_r = float(np.sum(f_zn_r[pellet] * depth_um[pellet]) / (np.sum(f_zn_r[pellet]) + 1e-8))
            w_p = float(np.sum(f_zn_p[pellet] * depth_um[pellet]) / (np.sum(f_zn_p[pellet]) + 1e-8))
            front_err_um = abs(w_p - w_r)
    else:
        w_r = float(np.sum(f_zn_r[pellet] * depth_um[pellet]) / (np.sum(f_zn_r[pellet]) + 1e-8))
        w_p = float(np.sum(f_zn_p[pellet] * depth_um[pellet]) / (np.sum(f_zn_p[pellet]) + 1e-8))
        front_err_um = abs(w_p - w_r)

    return {
        "in_pellet_na_zn_corr_pred": corr_pred,
        "in_pellet_na_zn_corr_ref": corr_ref,
        "in_pellet_na_zn_corr_abs_err": abs(corr_pred - corr_ref),
        "reaction_front_pos_err_um": float(front_err_um),
    }


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
    cutoff_cyc_per_px: float = 0.43,
    pixel_size_um: float = 25.0,
) -> Dict[str, float]:
    """Evaluate chemical super-resolution & fusion (revised Task T2 / legacy T3) with FRC band-limit & reaction front."""
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
        m_zn = _channel_25um_metrics(zn_p[iz], zn_r[iz], sz, prefix="zn_", cutoff_cyc_per_px=cutoff_cyc_per_px)
        m_na = _channel_25um_metrics(na_p[iz], na_r[iz], sn, prefix="na_", cutoff_cyc_per_px=cutoff_cyc_per_px)
        m_rf = _slice_reaction_front_and_phase_corr(zn_p[iz], na_p[iz], zn_r[iz], na_r[iz], pixel_size_um=pixel_size_um)
        slice_metrics.append({**m_zn, **m_na, **m_rf})

    keys = list(slice_metrics[0].keys())
    out: Dict[str, float] = {k: float(np.mean([sm[k] for sm in slice_metrics])) for k in keys}
    out["mean_rmse"] = 0.5 * (out["zn_rmse"] + out["na_rmse"])
    out["mean_pearson_r"] = 0.5 * (out["zn_pearson_r"] + out["na_pearson_r"])
    out["mean_ssim"] = 0.5 * (out["zn_ssim"] + out["na_ssim"])
    out["mean_bandlimited_rmse"] = 0.5 * (out["zn_bandlimited_rmse"] + out["na_bandlimited_rmse"])
    out["mean_bandlimited_pearson_r"] = 0.5 * (out["zn_bandlimited_pearson_r"] + out["na_bandlimited_pearson_r"])
    if "zn_reduced_chi2" in out and "na_reduced_chi2" in out:
        out["mean_reduced_chi2"] = 0.5 * (out["zn_reduced_chi2"] + out["na_reduced_chi2"])
        out["mean_bandlimited_reduced_chi2"] = 0.5 * (
            out["zn_bandlimited_reduced_chi2"] + out["na_bandlimited_reduced_chi2"]
        )

    # Optional HR physical consistency diagnostics
    if hr_pred_zn is not None and hr_pred_na is not None:
        hz = np.asarray(hr_pred_zn, dtype=np.float64)
        hn = np.asarray(hr_pred_na, dtype=np.float64)
        if hr_pore_mask is not None and np.any(hr_pore_mask > 0):
            pm = hr_pore_mask > 0
            solid_scale = float(np.percentile(np.abs(hz) + np.abs(hn), 90.0)) + 1e-8
            out["t3b_pore_leakage_rel"] = float(np.mean(np.abs(hz[pm]) + np.abs(hn[pm])) / solid_scale)
            out["pore_leakage_rel"] = out["t3b_pore_leakage_rel"]
        if hr_solid_mask is not None and np.any(hr_solid_mask > 0):
            sm = hr_solid_mask > 0
            tot_solid = hz[sm] + hn[sm]
            out["t3b_solid_stoichiometry_cv"] = float(np.std(tot_solid) / (np.abs(np.mean(tot_solid)) + 1e-8))
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


compute_t2_fusion_metrics = compute_t3_fusion_metrics
