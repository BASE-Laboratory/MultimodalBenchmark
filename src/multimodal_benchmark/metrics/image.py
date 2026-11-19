"""3D XCT super-resolution metrics: cupping-invariant PSNR/SSIM, Edge MTF50, and 3D Fourier Shell Correlation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class FSC3DResult:
    frequencies_cyc_per_vox: np.ndarray
    fsc_curve: np.ndarray
    cutoff_0p5_cyc_per_vox: float
    cutoff_0p143_cyc_per_vox: float
    cutoff_0p5_cyc_per_um: Optional[float]
    spectral_auc: float


def affine_cupping_normalize(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: Optional[np.ndarray] = None,
    detrend_sigma_vox: Optional[float] = None,
) -> np.ndarray:
    """Align prediction intensity scale/offset to reference via masked OLS and optional low-frequency cupping removal."""
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if detrend_sigma_vox is not None and detrend_sigma_vox > 0:
        p = p - gaussian_filter(p, sigma=detrend_sigma_vox, mode="nearest") + np.mean(p)
    valid = (mask > 0) if mask is not None else np.ones(r.shape, dtype=bool)
    pv = p[valid]
    rv = r[valid]
    if pv.size < 4 or np.var(pv) < 1e-15:
        return p.astype(np.float32)
    cov = float(np.mean((pv - pv.mean()) * (rv - rv.mean())))
    slope = cov / (float(np.var(pv)) + 1e-15)
    intercept = float(rv.mean() - slope * pv.mean())
    return (slope * p + intercept).astype(np.float32)


def compute_psnr(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: Optional[np.ndarray] = None,
    percentile_range: tuple[float, float] = (0.5, 99.5),
) -> float:
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    valid = (mask > 0) if mask is not None else np.ones(r.shape, dtype=bool)
    diff = p[valid] - r[valid]
    mse = float(np.mean(diff * diff))
    if mse <= 1e-18:
        return 99.0
    lo, hi = np.percentile(r[valid], percentile_range)
    dyn = max(float(hi - lo), float(np.ptp(r[valid])), 1e-6)
    return float(10.0 * np.log10((dyn * dyn) / mse))


def compute_ssim_3d(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute slice-averaged 3D SSIM (or masked SSIM inside cylindrical non-cupping ROI)."""
    p = np.asarray(pred, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    lo, hi = np.percentile(r, [0.5, 99.5])
    dyn = max(float(hi - lo), float(np.ptp(r)), 1e-6)

    if r.ndim == 2:
        win = min(7, min(r.shape) if min(r.shape) % 2 == 1 else min(r.shape) - 1)
        _, ssim_map = structural_similarity(r, p, data_range=dyn, win_size=max(3, win), full=True)
        return float(np.mean(ssim_map[mask > 0])) if mask is not None else float(np.mean(ssim_map))

    min_yx = min(r.shape[1], r.shape[2])
    win = min(7, min_yx if min_yx % 2 == 1 else min_yx - 1)
    win = max(3, win)
    vals: List[float] = []
    for iz in range(r.shape[0]):
        _, smap = structural_similarity(r[iz], p[iz], data_range=dyn, win_size=win, full=True)
        if mask is not None:
            m = mask[iz] > 0
            if np.any(m):
                vals.append(float(np.mean(smap[m])))
        else:
            vals.append(float(np.mean(smap)))
    return float(np.mean(vals)) if vals else 0.0


def compute_edge_mtf50(pred: np.ndarray, ref: np.ndarray) -> float:
    """Estimate effective MTF50 (cycles/voxel) from the radially averaged gradient transfer ratio |FFT(grad(pred))| / |FFT(grad(ref))|."""
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    gp = np.hypot(sobel(p, axis=-1), sobel(p, axis=-2))
    gr = np.hypot(sobel(r, axis=-1), sobel(r, axis=-2))
    if gp.ndim == 3:
        gp = gp[gp.shape[0] // 2]
        gr = gr[gr.shape[0] // 2]

    ny, nx = gr.shape
    fp = np.abs(np.fft.fftshift(np.fft.fft2(gp)))
    fr = np.abs(np.fft.fftshift(np.fft.fft2(gr)))
    cy, cx = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    rad = np.rint(np.hypot(yy - cy, xx - cx)).astype(np.int32)
    max_r = min(cy, cx)

    freqs: List[float] = []
    ratios: List[float] = []
    for ir in range(1, max_r + 1):
        ring = rad == ir
        denom = float(np.mean(fr[ring]))
        if denom > 1e-12:
            freqs.append(float(ir) / float(min(ny, nx)))
            ratios.append(float(np.mean(fp[ring]) / denom))

    if not freqs:
        return 0.0
    f_arr = np.asarray(freqs)
    mtf = np.clip(np.asarray(ratios) / max(ratios[0], 1e-6), 0.0, 1.5)
    below = np.where(mtf < 0.5)[0]
    if below.size == 0:
        return float(f_arr[-1])
    idx = int(below[0])
    if idx == 0:
        return float(f_arr[0])
    t = (0.5 - mtf[idx - 1]) / (mtf[idx] - mtf[idx - 1] - 1e-12)
    return float(f_arr[idx - 1] + np.clip(t, 0.0, 1.0) * (f_arr[idx] - f_arr[idx - 1]))


def compute_3d_fsc(
    pred: np.ndarray,
    ref: np.ndarray,
    voxel_size_um: Optional[float] = None,
) -> FSC3DResult:
    """Compute 3D Fourier Shell Correlation (FSC) between SR prediction and HR ground-truth volume."""
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError(f"Expected matching 3D volumes, got {a.shape} and {b.shape}")

    nz, ny, nx = a.shape
    wz = np.hanning(nz)[:, None, None]
    wy = np.hanning(ny)[None, :, None]
    wx = np.hanning(nx)[None, None, :]
    win = wz * wy * wx

    fa = np.fft.fftshift(np.fft.fftn((a - a.mean()) * win))
    fb = np.fft.fftshift(np.fft.fftn((b - b.mean()) * win))

    cz, cy, cx = nz // 2, ny // 2, nx // 2
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    min_n = float(min(nz, ny, nx))
    shells = np.rint(np.sqrt(((zz - cz) * (min_n / nz)) ** 2 + ((yy - cy) * (min_n / ny)) ** 2 + ((xx - cx) * (min_n / nx)) ** 2)).astype(np.int32)
    max_s = int(min_n // 2)

    num_prod = np.real(fa * np.conj(fb))
    pa = np.abs(fa) ** 2
    pb = np.abs(fb) ** 2

    freqs: List[float] = []
    fsc_vals: List[float] = []
    for s in range(1, max_s + 1):
        shell = shells == s
        if np.count_nonzero(shell) < 6:
            continue
        denom = np.sqrt(np.sum(pa[shell]) * np.sum(pb[shell])) + 1e-15
        fsc_vals.append(float(np.sum(num_prod[shell]) / denom))
        freqs.append(float(s) / min_n)

    f_arr = np.asarray(freqs, dtype=np.float64)
    c_arr = np.asarray(fsc_vals, dtype=np.float64)

    def _crossing(thresh: float) -> float:
        if f_arr.size == 0:
            return 0.0
        below = np.where(c_arr < thresh)[0]
        if below.size == 0:
            return float(f_arr[-1])
        idx = int(below[0])
        if idx == 0:
            return float(f_arr[0])
        t = (thresh - c_arr[idx - 1]) / (c_arr[idx] - c_arr[idx - 1] - 1e-12)
        return float(f_arr[idx - 1] + np.clip(t, 0.0, 1.0) * (f_arr[idx] - f_arr[idx - 1]))

    c05 = _crossing(0.5)
    c0143 = _crossing(0.143)
    auc = float(np.trapezoid(np.clip(c_arr, 0.0, 1.0), f_arr) / max(float(f_arr[-1] - f_arr[0]), 1e-6)) if f_arr.size >= 2 else 0.0
    c05_um = (c05 / voxel_size_um) if (voxel_size_um is not None and voxel_size_um > 0) else None
    return FSC3DResult(
        frequencies_cyc_per_vox=f_arr,
        fsc_curve=c_arr,
        cutoff_0p5_cyc_per_vox=c05,
        cutoff_0p143_cyc_per_vox=c0143,
        cutoff_0p5_cyc_per_um=c05_um,
        spectral_auc=auc,
    )


def compute_image_sr_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: Optional[np.ndarray] = None,
    voxel_size_um: Optional[float] = None,
) -> Dict[str, float]:
    """Compute standardized T1/T2 super-resolution image metrics."""
    pred_norm = affine_cupping_normalize(pred, ref, mask=mask)
    psnr_val = compute_psnr(pred_norm, ref, mask=mask)
    ssim_val = compute_ssim_3d(pred_norm, ref, mask=mask)
    mtf50_val = compute_edge_mtf50(pred_norm, ref)
    out = {
        "psnr_db": psnr_val,
        "ssim": ssim_val,
        "edge_mtf50_cyc_per_vox": mtf50_val,
    }
    if ref.ndim == 3:
        fsc = compute_3d_fsc(pred_norm, ref, voxel_size_um=voxel_size_um)
        out["fsc_0p5_cyc_per_vox"] = fsc.cutoff_0p5_cyc_per_vox
        out["fsc_0p143_cyc_per_vox"] = fsc.cutoff_0p143_cyc_per_vox
        out["fsc_auc"] = fsc.spectral_auc
        if fsc.cutoff_0p5_cyc_per_um is not None:
            out["fsc_0p5_cyc_per_um"] = fsc.cutoff_0p5_cyc_per_um
    return out
