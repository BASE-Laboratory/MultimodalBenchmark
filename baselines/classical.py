"""Classical mathematical baselines for T1/T2 super-resolution, T3 correlative fusion, and T4 segmentation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_opening, gaussian_filter, zoom
from scipy.optimize import nnls
from skimage.filters import threshold_multiotsu
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import downscale_local_mean, resize


# ---------------------------------------------------------------------------
# T1 & T2: 3D XCT Super-Resolution Baselines
# ---------------------------------------------------------------------------

def tricubic_upsample_3d(lr_zyx: np.ndarray, target_shape_zyx: Tuple[int, int, int]) -> np.ndarray:
    """3D tricubic spline interpolation (order=3)."""
    lr = np.asarray(lr_zyx, dtype=np.float32)
    factors = tuple(float(t) / float(s) for t, s in zip(target_shape_zyx, lr.shape))
    return zoom(lr, zoom=factors, order=3, mode="nearest", grid_mode=True).astype(np.float32)


def _lanczos_resample_1d_axis(arr: np.ndarray, target_len: int, axis: int, a: int = 3) -> np.ndarray:
    src_len = arr.shape[axis]
    if src_len == target_len:
        return arr.astype(np.float32, copy=False)

    # Coordinate mapping (grid-center aligned)
    scale = float(src_len) / float(target_len)
    out_coords = (np.arange(target_len, dtype=np.float64) + 0.5) * scale - 0.5
    base = np.floor(out_coords).astype(np.int32)

    weights = np.zeros((target_len, 2 * a), dtype=np.float64)
    indices = np.zeros((target_len, 2 * a), dtype=np.int32)
    for k_idx, offset in enumerate(range(-a + 1, a + 1)):
        pos = base + offset
        x = out_coords - pos
        # Lanczos kernel L(x) = sinc(x) * sinc(x / a)
        w = np.sinc(x) * np.sinc(x / float(a))
        w[np.abs(x) >= a] = 0.0
        weights[:, k_idx] = w
        indices[:, k_idx] = np.clip(pos, 0, src_len - 1)

    weights /= np.sum(weights, axis=1, keepdims=True) + 1e-12

    moved = np.moveaxis(arr, axis, 0)
    rest_shape = moved.shape[1:]
    flat = moved.reshape(src_len, -1)
    gathered = flat[indices]                                   # (target_len, 2*a, M)
    res = np.einsum("tk,tkm->tm", weights, gathered)           # (target_len, M)
    out_moved = res.reshape((target_len,) + rest_shape)
    return np.moveaxis(out_moved, 0, axis).astype(np.float32)


def lanczos_upsample_3d(lr_zyx: np.ndarray, target_shape_zyx: Tuple[int, int, int], lobes: int = 3) -> np.ndarray:
    """Separable 3D 3-lobe Lanczos windowed-sinc interpolation."""
    out = np.asarray(lr_zyx, dtype=np.float32)
    for ax, tgt in enumerate(target_shape_zyx):
        out = _lanczos_resample_1d_axis(out, int(tgt), axis=ax, a=lobes)
    return out


def tv_deconv_upsample_3d(
    lr_zyx: np.ndarray,
    target_shape_zyx: Tuple[int, int, int],
    n_iter: int = 12,
    psf_sigma_hr_vox: float = 1.0,
    tv_weight: float = 0.02,
    step_size: float = 0.6,
) -> np.ndarray:
    """Iterative back-projection + Total Variation regularization for 3D XCT super-resolution."""
    lr = np.asarray(lr_zyx, dtype=np.float32)
    est = tricubic_upsample_3d(lr, target_shape_zyx)
    dyn = max(float(np.ptp(lr)), 1e-6)
    inv_factors = tuple(float(s) / float(t) for s, t in zip(lr.shape, target_shape_zyx))

    for _ in range(n_iter):
        blurred = gaussian_filter(est, sigma=psf_sigma_hr_vox, mode="nearest")
        sim_lr = zoom(blurred, zoom=inv_factors, order=1, mode="nearest", grid_mode=True)
        resid_hr = tricubic_upsample_3d(lr - sim_lr, target_shape_zyx)
        est = est + step_size * gaussian_filter(resid_hr, sigma=psf_sigma_hr_vox, mode="nearest")
        if tv_weight > 0:
            est = denoise_tv_chambolle(est, weight=tv_weight * dyn, max_num_iter=10).astype(np.float32)
    return est.astype(np.float32)


# ---------------------------------------------------------------------------
# T3: Correlative Multimodal Chemical Fusion Baselines (50 um + XCT -> 25 um)
# ---------------------------------------------------------------------------

def bicubic_chemical_upsample(
    chem_50um_yx: np.ndarray,
    target_shape_yx: Tuple[int, int] = (40, 40),
) -> np.ndarray:
    """Pure 2D bicubic upsampling without structural XCT guidance (non-negative clamped)."""
    up = resize(
        np.asarray(chem_50um_yx, dtype=np.float32),
        target_shape_yx,
        order=3,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )
    return np.clip(up, 0.0, None).astype(np.float32)


def joint_bilateral_upsample(
    chem_50um_yx: np.ndarray,
    xct_guide_25um_yx: np.ndarray,
    sigma_spatial_hr_px: float = 1.6,
    sigma_range_rel: float = 0.18,
    window_radius: int = 3,
    pore_mask_yx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Joint Bilateral Upsampling (JBU) of 50 um XRD-CT guided by co-registered DIAD XCT."""
    c_lr = np.asarray(chem_50um_yx, dtype=np.float64)
    guide = np.asarray(xct_guide_25um_yx, dtype=np.float64)
    ny_hr, nx_hr = guide.shape
    c_init = resize(c_lr, (ny_hr, nx_hr), order=1, mode="edge", preserve_range=True)

    sigma_r = max(float(np.std(guide)) * sigma_range_rel, 1e-6)
    num = np.zeros_like(guide)
    den = np.zeros_like(guide)

    for dy in range(-window_radius, window_radius + 1):
        for dx in range(-window_radius, window_radius + 1):
            w_spatial = np.exp(-0.5 * (dy * dy + dx * dx) / (sigma_spatial_hr_px ** 2))
            g_shift = np.pad(guide, ((abs(dy), abs(dy)), (abs(dx), abs(dx))), mode="edge")
            c_shift = np.pad(c_init, ((abs(dy), abs(dy)), (abs(dx), abs(dx))), mode="edge")
            y0 = abs(dy) + dy
            x0 = abs(dx) + dx
            g_nb = g_shift[y0 : y0 + ny_hr, x0 : x0 + nx_hr]
            c_nb = c_shift[y0 : y0 + ny_hr, x0 : x0 + nx_hr]
            w_range = np.exp(-0.5 * ((guide - g_nb) / sigma_r) ** 2)
            w = w_spatial * w_range
            num += w * c_nb
            den += w

    out = np.clip(num / (den + 1e-12), 0.0, None)
    if pore_mask_yx is not None:
        out[pore_mask_yx > 0] = 0.0
    return out.astype(np.float32)


def guided_tv_chemical_fusion(
    chem_50um_yx: np.ndarray,
    xct_guide_25um_yx: np.ndarray,
    tv_weight: float = 0.08,
    edge_sharpness: float = 0.15,
    n_iter: int = 80,
    step_size: float = 0.35,
    pore_mask_yx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Structure-guided Total Variation super-resolution enforcing exact 50 um forward-PSF conservation."""
    c_lr = np.asarray(chem_50um_yx, dtype=np.float64)
    guide = np.asarray(xct_guide_25um_yx, dtype=np.float64)
    ny_hr, nx_hr = guide.shape
    fy = ny_hr // c_lr.shape[0]
    fx = nx_hr // c_lr.shape[1]

    u = resize(c_lr, (ny_hr, nx_hr), order=1, mode="edge", preserve_range=True)
    gy, gx = np.gradient(guide)
    grad_mag2 = gy * gy + gx * gx
    sigma_g2 = max(float(np.mean(grad_mag2)) * (edge_sharpness ** 2) * 10.0, 1e-10)
    edge_w = np.exp(-grad_mag2 / sigma_g2)

    eps = 1e-4 * max(float(np.ptp(c_lr)), 1.0)
    for _ in range(n_iter):
        # Data fidelity gradient: D^T (D u - c_lr)
        sim_lr = downscale_local_mean(u, (fy, fx))
        fid_grad = np.repeat(np.repeat(sim_lr - c_lr, fy, axis=0), fx, axis=1)

        # Anisotropic structure-guided TV gradient: -div(edge_w * grad(u) / |grad(u)|)
        uy, ux = np.gradient(u)
        norm_u = np.sqrt(uy * uy + ux * ux + eps * eps)
        py = edge_w * (uy / norm_u)
        px = edge_w * (ux / norm_u)
        div_p = np.gradient(py, axis=0) + np.gradient(px, axis=1)

        u = np.clip(u - step_size * (fid_grad - tv_weight * div_p), 0.0, None)
        if pore_mask_yx is not None:
            u[pore_mask_yx > 0] = 0.0

    return u.astype(np.float32)


def cnmf_coupled_fusion(
    zn_50um_yx: np.ndarray,
    na_50um_yx: np.ndarray,
    xct_guide_25um_yx: np.ndarray,
    pore_mask_yx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coupled Non-Negative Matrix Factorization / NNLS abundance sharpening guided by XCT solid fraction."""
    zn_jbu = joint_bilateral_upsample(zn_50um_yx, xct_guide_25um_yx, pore_mask_yx=pore_mask_yx)
    na_jbu = joint_bilateral_upsample(na_50um_yx, xct_guide_25um_yx, pore_mask_yx=pore_mask_yx)

    g = np.asarray(xct_guide_25um_yx, dtype=np.float64)
    g_pos = np.clip(g - np.percentile(g, 5.0), 0.0, None)
    fy = g.shape[0] // zn_50um_yx.shape[0]
    fx = g.shape[1] // zn_50um_yx.shape[1]
    g_lr = downscale_local_mean(g_pos, (fy, fx))

    # Fit linear attenuation model g_lr ~ a_zn * C_zn + a_na * C_na via NNLS
    design = np.column_stack([zn_50um_yx.ravel(), na_50um_yx.ravel()]).astype(np.float64)
    sol, _ = nnls(design, g_lr.ravel())
    pred_tot = sol[0] * zn_jbu + sol[1] * na_jbu + 1e-8
    mod = np.clip(g_pos / np.maximum(pred_tot, np.percentile(g_pos, 15.0) + 1e-6), 0.25, 2.5)

    zn_out = zn_jbu * mod
    na_out = na_jbu * mod
    # Re-normalize each 2x2 macro-pixel to conserve coarse 50 um integral
    for arr_hr, arr_lr in ((zn_out, zn_50um_yx), (na_out, na_50um_yx)):
        sim = downscale_local_mean(arr_hr, (fy, fx)) + 1e-8
        ratio = np.repeat(np.repeat(np.clip(arr_lr, 0.0, None) / sim, fy, axis=0), fx, axis=1)
        arr_hr *= ratio
        if pore_mask_yx is not None:
            arr_hr[pore_mask_yx > 0] = 0.0

    return zn_out.astype(np.float32), na_out.astype(np.float32)


def guided_filter_chemical_upsample(
    chem_50um_yx: np.ndarray,
    xct_guide_25um_yx: np.ndarray,
    radius_px: int = 2,
    eps_rel: float = 0.02,
    pore_mask_yx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """He et al. (2013) edge-preserving Guided Image Filter upsampling for Task T2 chemical fusion."""
    from scipy.ndimage import uniform_filter

    c_lr = np.asarray(chem_50um_yx, dtype=np.float64)
    guide = np.asarray(xct_guide_25um_yx, dtype=np.float64)
    ny_hr, nx_hr = guide.shape
    p = resize(c_lr, (ny_hr, nx_hr), order=1, mode="edge", preserve_range=True)

    win = 2 * int(radius_px) + 1
    eps = (float(eps_rel) * max(float(np.ptp(guide)), 1e-6)) ** 2 + 1e-10

    mean_i = uniform_filter(guide, size=win, mode="reflect")
    mean_p = uniform_filter(p, size=win, mode="reflect")
    corr_i = uniform_filter(guide * guide, size=win, mode="reflect")
    corr_ip = uniform_filter(guide * p, size=win, mode="reflect")

    var_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = uniform_filter(a, size=win, mode="reflect")
    mean_b = uniform_filter(b, size=win, mode="reflect")
    out = np.clip(mean_a * guide + mean_b, 0.0, None)
    if pore_mask_yx is not None:
        out[pore_mask_yx > 0] = 0.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Optional T4: Sparse-View 25 um XRD-CT Sinogram Reconstruction Baselines
# ---------------------------------------------------------------------------

def fbp_sparse_reconstruct_2d(
    sinogram_at: np.ndarray,
    angles_deg: np.ndarray,
    center_px: Optional[float] = None,
    filter_name: str = "hann",
) -> np.ndarray:
    """Filtered Back-Projection (FBP) baseline for subsampled sparse-view XRD-CT sinograms."""
    from registration.xrdct_reference import reconstruct_sinogram_2d

    return reconstruct_sinogram_2d(sinogram_at, angles_deg, center_px=center_px, filter_name=filter_name)


def sirt_reconstruct_2d(
    sinogram_at: np.ndarray,
    angles_deg: np.ndarray,
    center_px: Optional[float] = None,
    n_iter: int = 60,
    relaxation: float = 0.85,
    non_negative: bool = True,
) -> np.ndarray:
    """Simultaneous Iterative Reconstruction Technique (SIRT) with non-negativity for Optional Task T4."""
    from scipy.ndimage import shift as nd_shift
    from skimage.transform import iradon, radon

    sino = np.asarray(sinogram_at, dtype=np.float64).copy()
    ang = np.asarray(angles_deg, dtype=np.float64).copy()
    if ang.size > 2 and abs(abs(float(ang[-1] - ang[0])) - 360.0) < 0.35:
        sino = sino[:-1]
        ang = ang[:-1]

    n_ang, n_trans = sino.shape
    nominal_fbp = float(n_trans // 2)
    if center_px is not None:
        delta = float(center_px) - nominal_fbp
        if abs(delta) > 1e-6:
            sino = nd_shift(sino, shift=(0.0, -delta), order=3, mode="nearest")

    # Precompute row (ray length R*1) and column (backprojection R^T*1) diagonal normalizers for exact Landweber/SIRT stability
    ones_img = np.ones((n_trans, n_trans), dtype=np.float64)
    yy, xx = np.ogrid[:n_trans, :n_trans]
    circ_mask = ((yy - 0.5 * (n_trans - 1)) ** 2 + (xx - 0.5 * (n_trans - 1)) ** 2) <= (0.5 * n_trans) ** 2
    ones_img *= circ_mask

    ray_sums = np.maximum(radon(ones_img, theta=ang, circle=True), 1.0)  # (n_trans, n_ang)
    col_sums = np.maximum(
        iradon(np.ones((n_trans, n_ang), dtype=np.float64), theta=ang, output_size=n_trans, filter_name=None, circle=True),
        1.0,
    )

    span = float(np.ptp(ang)) if n_ang > 1 else 180.0
    scale = 0.5 if span > 200.0 else 1.0
    x = np.zeros((n_trans, n_trans), dtype=np.float64)
    sino_t = sino.T  # (n_trans, n_ang) matching skimage.transform.radon convention
    for _ in range(int(n_iter)):
        fwd = radon(x, theta=ang, circle=True)
        resid = (sino_t - fwd) / ray_sums
        bp = iradon(resid, theta=ang, output_size=n_trans, filter_name=None, circle=True)
        x = x + float(relaxation) * (bp / col_sums) * circ_mask
        if non_negative:
            x = np.maximum(x, 0.0)
    return (x * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Auxiliary 3-Phase XCT Segmentation Helper (Used in Registration & Masks)
# ---------------------------------------------------------------------------

def multi_otsu_watershed_3d(
    xct_zyx: np.ndarray,
    num_classes: int = 3,
    smooth_sigma_vox: float = 0.8,
) -> np.ndarray:
    """Smooth + 3-class Multi-Otsu thresholding + morphological speckle cleanup."""
    sm = gaussian_filter(np.asarray(xct_zyx, dtype=np.float32), sigma=smooth_sigma_vox)
    thresholds = threshold_multiotsu(sm, classes=num_classes)
    labels = np.digitize(sm, bins=thresholds).astype(np.int32)
    if num_classes >= 3:
        top = binary_opening(labels == (num_classes - 1), iterations=1)
        labels[(labels == (num_classes - 1)) & (~top)] = num_classes - 2
    return labels
