"""Stage 0A (Part 2): Correlative DIAD XCT <-> XRD-CT chirality, axial z-height, and 2D similarity alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, uniform_filter1d
from scipy.optimize import minimize


@dataclass(frozen=True)
class CorrelativeAlignmentResult:
    best_z_center_vox: float
    best_z_center_um: float
    flip_x: bool
    shift_yx_xrd_px: Tuple[float, float]
    shift_yx_um: Tuple[float, float]
    rotation_deg: float
    scale: float
    nmi_score: float
    pearson_r: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalized_mutual_information_2d(
    img_a: np.ndarray,
    img_b: np.ndarray,
    bins: int = 32,
) -> float:
    """Compute Normalized Mutual Information NMI(A, B) = (H(A) + H(B)) / H(A, B) >= 1.0."""
    a = np.asarray(img_a, dtype=np.float64).ravel()
    b = np.asarray(img_b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 16 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return 1.0

    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist_2d / (np.sum(hist_2d) + 1e-15)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    nz_x = px > 0
    nz_y = py > 0
    nz_xy = pxy > 0
    hx = -np.sum(px[nz_x] * np.log(px[nz_x]))
    hy = -np.sum(py[nz_y] * np.log(py[nz_y]))
    hxy = -np.sum(pxy[nz_xy] * np.log(pxy[nz_xy]))
    return float((hx + hy) / (hxy + 1e-15))


def blur_xct_to_xrdct_grid(
    xct_zyx: np.ndarray,
    xct_voxel_um: float = 0.54,
    xrd_pixel_um: float = 25.0,
    beam_thickness_um: Optional[float] = None,
    target_yx_shape: Optional[Tuple[int, int]] = None,
    fov_center_yx_vox: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Apply physical 3D pencil-beam PSF (axial slab thickness + in-plane spot blur) and resample onto XRD-CT yx grid.

    For DIAD 0.54 um XCT:
      - 25 um XRD-CT averages over ~46.3 axial XCT slices per slice.
      - 50 um XRD-CT averages over ~92.6 axial XCT slices per slice.
    """
    vol = np.asarray(xct_zyx, dtype=np.float32)
    nz, ny, nx = vol.shape
    slab_um = float(beam_thickness_um if beam_thickness_um is not None else xrd_pixel_um)
    slab_vox = max(1, int(np.rint(slab_um / xct_voxel_um)))
    inplane_sigma_vox = max(0.5, (xrd_pixel_um / xct_voxel_um) / 2.355)

    # 1. Axial boxcar integration along z (pencil-beam vertical height)
    if slab_vox > 1 and nz > 1:
        vol_z = uniform_filter1d(vol, size=min(slab_vox, nz), axis=0, mode="nearest")
    else:
        vol_z = vol

    # 2. Sample onto target (ny_xrd, nx_xrd) grid at physical step = xrd_pixel_um / xct_voxel_um
    step_vox = xrd_pixel_um / xct_voxel_um
    if target_yx_shape is None:
        ny_out = max(4, int(np.floor(ny / step_vox)))
        nx_out = max(4, int(np.floor(nx / step_vox)))
    else:
        ny_out, nx_out = target_yx_shape

    if fov_center_yx_vox is None:
        cy, cx = 0.5 * (ny - 1), 0.5 * (nx - 1)
    else:
        cy, cx = fov_center_yx_vox

    out = np.zeros((nz, ny_out, nx_out), dtype=np.float32)
    offs_y = (np.arange(ny_out, dtype=np.float64) - 0.5 * (ny_out - 1)) * step_vox + cy
    offs_x = (np.arange(nx_out, dtype=np.float64) - 0.5 * (nx_out - 1)) * step_vox + cx

    # Pre-filter in-plane slice-by-slice and bilinearly sample onto coarse grid
    yi0 = np.clip(np.floor(offs_y).astype(np.int32), 0, ny - 1)
    yi1 = np.clip(yi0 + 1, 0, ny - 1)
    wy = (offs_y - yi0).astype(np.float32)[:, None]

    xi0 = np.clip(np.floor(offs_x).astype(np.int32), 0, nx - 1)
    xi1 = np.clip(xi0 + 1, 0, nx - 1)
    wx = (offs_x - xi0).astype(np.float32)[None, :]

    for iz in range(nz):
        sl = gaussian_filter(vol_z[iz], sigma=inplane_sigma_vox, mode="nearest")
        v00 = sl[np.ix_(yi0, xi0)]
        v01 = sl[np.ix_(yi0, xi1)]
        v10 = sl[np.ix_(yi1, xi0)]
        v11 = sl[np.ix_(yi1, xi1)]
        out[iz] = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)

    return out


def _warp_2d_similarity(
    img: np.ndarray,
    dy_px: float,
    dx_px: float,
    rot_deg: float,
    scale: float,
) -> np.ndarray:
    """Apply 2D similarity transform around image center mapping output coords -> input coords."""
    ny, nx = img.shape
    center = np.array([0.5 * (ny - 1), 0.5 * (nx - 1)], dtype=np.float64)
    theta = np.deg2rad(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    inv_s = 1.0 / max(scale, 1e-3)
    # Output coord y_out = center + scale * R * (y_in - center) + shift => y_in = center + inv_s * R^T * (y_out - shift - center)
    mat = inv_s * np.array([[c, s], [-s, c]], dtype=np.float64)
    shift = np.array([dy_px, dx_px], dtype=np.float64)
    offset = center - mat @ (center + shift)
    return affine_transform(img, matrix=mat, offset=offset, order=1, mode="nearest")


def align_xrdct_slice_to_diad_xct(
    xrd_total_yx: np.ndarray,
    blurred_xct_zyx: np.ndarray,
    xct_voxel_um: float = 0.54,
    xrd_pixel_um: float = 25.0,
    z_search_range: Optional[Tuple[int, int]] = None,
    test_both_chiralities: bool = True,
) -> CorrelativeAlignmentResult:
    """Register a 2D total-scattering XRD-CT slice to an anisotropically PSF-blurred DIAD XCT stack.

    Explicitly tests both x-chiralities (flip_x in {False, True}) to resolve the Savu AstraReconGpu
    vs TomoPy gridrec mirror convention, finds the optimal axial z-height, and refines sub-pixel
    2D similarity (dy, dx, rotation_deg, scale) via Normalized Mutual Information.
    """
    target = np.asarray(xrd_total_yx, dtype=np.float64)
    stack = np.asarray(blurred_xct_zyx, dtype=np.float64)
    if stack.shape[1:] != target.shape:
        raise ValueError(f"Blurred XCT yx shape {stack.shape[1:]} must match XRD-CT slice {target.shape}")

    nz = stack.shape[0]
    z0, z1 = z_search_range if z_search_range is not None else (0, nz)
    z0 = max(0, int(z0))
    z1 = min(nz, int(z1))
    chiralities = (False, True) if test_both_chiralities else (False,)

    best_coarse = (-1.0, z0, False, 0, 0)
    # Coarse integer shift + axial z + chirality sweep
    max_int_shift = max(2, min(5, target.shape[0] // 5))
    for flip_x in chiralities:
        for iz in range(z0, z1):
            cand = np.fliplr(stack[iz]) if flip_x else stack[iz]
            for dy in range(-max_int_shift, max_int_shift + 1):
                for dx in range(-max_int_shift, max_int_shift + 1):
                    moved = _warp_2d_similarity(cand, float(dy), float(dx), 0.0, 1.0)
                    score = normalized_mutual_information_2d(target, moved, bins=24)
                    if score > best_coarse[0]:
                        best_coarse = (score, iz, flip_x, dy, dx)

    _, z_init, best_flip, dy_init, dx_init = best_coarse

    def _interp_z_slice(z_float: float) -> np.ndarray:
        zc = float(np.clip(z_float, 0.0, nz - 1.0))
        i0 = int(np.floor(zc))
        i1 = min(nz - 1, i0 + 1)
        w = zc - i0
        sl = (1.0 - w) * stack[i0] + w * stack[i1]
        return np.fliplr(sl) if best_flip else sl

    # Continuous Powell refinement over (z, dy, dx, rot_deg, scale)
    x0 = np.array([float(z_init), float(dy_init), float(dx_init), 0.0, 1.0], dtype=np.float64)
    bounds = [
        (float(max(0, z_init - 5)), float(min(nz - 1, z_init + 5))),
        (dy_init - 3.0, dy_init + 3.0),
        (dx_init - 3.0, dx_init + 3.0),
        (-8.0, 8.0),
        (0.92, 1.08),
    ]

    def _obj(params: np.ndarray) -> float:
        zf, dyf, dxf, rotf, scf = params
        if not (bounds[0][0] <= zf <= bounds[0][1] and bounds[4][0] <= scf <= bounds[4][1]):
            return 10.0
        sl = _interp_z_slice(zf)
        warped = _warp_2d_similarity(sl, dyf, dxf, rotf, scf)
        # Combine NMI with slight correlation tie-breaker for smooth sub-pixel convergence
        nmi = normalized_mutual_information_2d(target, warped, bins=24)
        if np.std(warped) > 1e-12 and np.std(target) > 1e-12:
            r = float(np.corrcoef(target.ravel(), warped.ravel())[0, 1])
        else:
            r = 0.0
        return -(nmi + 0.15 * r)

    opt = minimize(_obj, x0, method="Powell", bounds=bounds, options={"maxiter": 250, "ftol": 1e-4})
    z_opt, dy_opt, dx_opt, rot_opt, sc_opt = [float(v) for v in opt.x]
    final_slice = _warp_2d_similarity(_interp_z_slice(z_opt), dy_opt, dx_opt, rot_opt, sc_opt)
    final_nmi = normalized_mutual_information_2d(target, final_slice, bins=24)
    final_r = float(np.corrcoef(target.ravel(), final_slice.ravel())[0, 1]) if np.std(final_slice) > 1e-12 else 0.0

    return CorrelativeAlignmentResult(
        best_z_center_vox=z_opt,
        best_z_center_um=z_opt * xct_voxel_um,
        flip_x=best_flip,
        shift_yx_xrd_px=(dy_opt, dx_opt),
        shift_yx_um=(dy_opt * xrd_pixel_um, dx_opt * xrd_pixel_um),
        rotation_deg=rot_opt,
        scale=sc_opt,
        nmi_score=final_nmi,
        pearson_r=final_r,
    )
