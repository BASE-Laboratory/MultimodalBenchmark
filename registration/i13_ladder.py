"""Stage 0B: I13-2 multi-magnification 7-DOF similarity registration, landmark TRE, and local-tomography mask."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from basecoat import (
    AxisSpec,
    SimilarityTransform3D as BasecoatSimilarityTransform3D,
    Volume,
    resample_volume,
)
from scipy.ndimage import center_of_mass, gaussian_filter, label as nd_label, map_coordinates, sobel
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu


def _require_sitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for multi-resolution NMI/MSE registration fitting; "
            "install via `pip install multimodal-benchmark[registration]`."
        ) from exc
    return sitk


def _rotmat_xyz_to_versor_xyzw(r_xyz: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert an orthonormal right-handed 3x3 XYZ rotation matrix to an ITK unit versor (qx, qy, qz, qw)."""
    tr = float(r_xyz[0, 0] + r_xyz[1, 1] + r_xyz[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r_xyz[2, 1] - r_xyz[1, 2]) / s
        qy = (r_xyz[0, 2] - r_xyz[2, 0]) / s
        qz = (r_xyz[1, 0] - r_xyz[0, 1]) / s
    elif r_xyz[0, 0] > r_xyz[1, 1] and r_xyz[0, 0] > r_xyz[2, 2]:
        s = math.sqrt(1.0 + r_xyz[0, 0] - r_xyz[1, 1] - r_xyz[2, 2]) * 2.0
        qw = (r_xyz[2, 1] - r_xyz[1, 2]) / s
        qx = 0.25 * s
        qy = (r_xyz[0, 1] + r_xyz[1, 0]) / s
        qz = (r_xyz[0, 2] + r_xyz[2, 0]) / s
    elif r_xyz[1, 1] > r_xyz[2, 2]:
        s = math.sqrt(1.0 + r_xyz[1, 1] - r_xyz[0, 0] - r_xyz[2, 2]) * 2.0
        qw = (r_xyz[0, 2] - r_xyz[2, 0]) / s
        qx = (r_xyz[0, 1] + r_xyz[1, 0]) / s
        qy = 0.25 * s
        qz = (r_xyz[1, 2] + r_xyz[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r_xyz[2, 2] - r_xyz[0, 0] - r_xyz[1, 1]) * 2.0
        qw = (r_xyz[1, 0] - r_xyz[0, 1]) / s
        qx = (r_xyz[0, 2] + r_xyz[2, 0]) / s
        qy = (r_xyz[1, 2] + r_xyz[2, 1]) / s
        qz = 0.25 * s
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    nrm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return (float(qx / nrm), float(qy / nrm), float(qz / nrm), float(qw / nrm))


@dataclass(frozen=True)
class SimilarityTransform3D:
    """3D 7-DOF similarity transform DTO in physical micron coordinates (SimpleITK XYZ schema, backed by basecoat)."""

    matrix_3x3: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    translation_xyz_um: Tuple[float, float, float]
    center_xyz_um: Tuple[float, float, float]
    scale: float
    versor_xyzw: Tuple[float, float, float, float]

    def to_basecoat(
        self,
        from_space: str = "fixed",
        to_space: str = "moving",
    ) -> BasecoatSimilarityTransform3D:
        """Convert to a `basecoat.SimilarityTransform3D` in canonical `('z', 'y', 'x')` order."""
        scaled_xyz = np.asarray(self.matrix_3x3, dtype=np.float64)
        r_xyz = scaled_xyz / float(self.scale)
        r_zyx = r_xyz[::-1, ::-1]
        cx, cy, cz = self.center_xyz_um
        tx, ty, tz = self.translation_xyz_um
        return BasecoatSimilarityTransform3D(
            rotation_zyx=r_zyx,
            scale=float(self.scale),
            center_zyx=(float(cz), float(cy), float(cx)),
            translation_zyx=(float(tz), float(ty), float(tx)),
            from_space=from_space,
            to_space=to_space,
            unit="um",
        )

    @classmethod
    def from_basecoat(cls, tx: BasecoatSimilarityTransform3D) -> "SimilarityTransform3D":
        """Construct from a `basecoat.SimilarityTransform3D` without requiring SimpleITK."""
        r_xyz = np.asarray(tx.rotation_zyx, dtype=np.float64)[::-1, ::-1]
        scaled_xyz = float(tx.scale) * r_xyz
        cz, cy, cx = tx.center_zyx
        tz, ty, tx_um = tx.translation_zyx
        return cls(
            matrix_3x3=tuple(tuple(float(v) for v in row) for row in scaled_xyz),  # type: ignore[arg-type]
            translation_xyz_um=(float(tx_um), float(ty), float(tz)),
            center_xyz_um=(float(cx), float(cy), float(cz)),
            scale=float(tx.scale),
            versor_xyzw=_rotmat_xyz_to_versor_xyzw(r_xyz),
        )

    def transform_points_xyz_um(self, pts_xyz_um: np.ndarray) -> np.ndarray:
        """Map fixed physical points (N, 3) in (x, y, z) um to moving physical points via basecoat."""
        pts_xyz = np.asarray(pts_xyz_um, dtype=np.float64)
        mapped_zyx = self.to_basecoat().transform_points(pts_xyz[..., ::-1])
        return mapped_zyx[..., ::-1]

    def to_sitk(self) -> Any:
        return self.to_basecoat().to_sitk()

    @classmethod
    def from_sitk(cls, tx: Any) -> "SimilarityTransform3D":
        bc = BasecoatSimilarityTransform3D.from_sitk(tx, from_space="fixed", to_space="moving", unit="um")
        return cls.from_basecoat(bc)

    def save_json(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> "SimilarityTransform3D":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(
            matrix_3x3=tuple(tuple(float(v) for v in row) for row in raw["matrix_3x3"]),
            translation_xyz_um=tuple(float(v) for v in raw["translation_xyz_um"]),
            center_xyz_um=tuple(float(v) for v in raw["center_xyz_um"]),
            scale=float(raw["scale"]),
            versor_xyzw=tuple(float(v) for v in raw["versor_xyzw"]),
        )


@dataclass(frozen=True)
class LadderRegistrationResult:
    transform: SimilarityTransform3D
    final_metric_value: float
    stop_condition: str
    tre_mean_um: float
    tre_median_um: float
    tre_p95_um: float
    num_matched_landmarks: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def numpy_zyx_to_sitk(
    vol_zyx: np.ndarray,
    spacing_zyx_um: Tuple[float, float, float],
    origin_zyx_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    """Convert a (Z, Y, X) NumPy volume to a SimpleITK Image with physical (X, Y, Z) spacing in um."""
    sitk = _require_sitk()
    img = sitk.GetImageFromArray(np.asarray(vol_zyx, dtype=np.float32))
    sz, sy, sx = spacing_zyx_um
    oz, oy, ox = origin_zyx_um
    img.SetSpacing((float(sx), float(sy), float(sz)))
    img.SetOrigin((float(ox), float(oy), float(oz)))
    return img


def resample_moving_to_fixed_zyx(
    moving_zyx: np.ndarray,
    moving_spacing_zyx_um: Tuple[float, float, float],
    fixed_shape_zyx: Tuple[int, int, int],
    fixed_spacing_zyx_um: Tuple[float, float, float],
    transform: SimilarityTransform3D | BasecoatSimilarityTransform3D,
    moving_origin_zyx_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_origin_zyx_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    interpolator: Optional[int] = None,
) -> np.ndarray:
    """Resample moving 3D volume onto fixed physical grid using `basecoat.resample_volume`."""
    bc_tx = (
        transform
        if isinstance(transform, BasecoatSimilarityTransform3D)
        else transform.to_basecoat(from_space="fixed", to_space="moving")
    )
    moving_axes = tuple(
        AxisSpec(name=ax, kind="space", unit="um", scale=float(sp), offset=float(orig))
        for ax, sp, orig in zip(("z", "y", "x"), moving_spacing_zyx_um, moving_origin_zyx_um)
    )
    fixed_axes = tuple(
        AxisSpec(name=ax, kind="space", unit="um", scale=float(sp), offset=float(orig))
        for ax, sp, orig in zip(("z", "y", "x"), fixed_spacing_zyx_um, fixed_origin_zyx_um)
    )
    # Map SimpleITK interpolator enum constants (sitkNearestNeighbor == 1, sitkLinear == 2, sitkBSpline == 3)
    if interpolator is None or interpolator == 2:
        spline_order = 1
    elif interpolator == 1:
        spline_order = 0
    else:
        spline_order = 3

    mov_vol = Volume(np.asarray(moving_zyx, dtype=np.float32), axes=moving_axes, space=bc_tx.to_space)
    out_vol = resample_volume(
        mov_vol,
        fixed_axes,
        target_shape=(int(fixed_shape_zyx[0]), int(fixed_shape_zyx[1]), int(fixed_shape_zyx[2])),
        transform=bc_tx,
        order=spline_order,
    )
    return np.asarray(out_vol.data, dtype=np.float32)


def extract_feature_centroids_xyz_um(
    vol_zyx: np.ndarray,
    spacing_zyx_um: Tuple[float, float, float],
    origin_zyx_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_voxels: int = 12,
    max_voxels: int = 50000,
    bright_features: bool = True,
) -> np.ndarray:
    """Extract 3D physical centroids (X, Y, Z) in um of isolated beads/inclusions or closed macropores."""
    vol = np.asarray(vol_zyx, dtype=np.float32)
    if np.ptp(vol) < 1e-8:
        return np.zeros((0, 3), dtype=np.float64)

    thresh = float(threshold_otsu(vol))
    binary = (vol > thresh) if bright_features else (vol < thresh)
    labeled, n_comp = nd_label(binary)
    if n_comp == 0:
        return np.zeros((0, 3), dtype=np.float64)

    counts = np.bincount(labeled.ravel())
    valid_labels = np.where((counts >= min_voxels) & (counts <= max_voxels))[0]
    valid_labels = valid_labels[valid_labels > 0]
    if valid_labels.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    sz, sy, sx = spacing_zyx_um
    oz, oy, ox = origin_zyx_um
    coms_zyx = center_of_mass(binary, labeled, valid_labels[:250])
    if isinstance(coms_zyx, tuple):
        coms_zyx = [coms_zyx]
    centroids: List[Tuple[float, float, float]] = [
        (ox + float(cx) * sx, oy + float(cy) * sy, oz + float(cz) * sz)
        for cz, cy, cx in coms_zyx
    ]
    return np.asarray(centroids, dtype=np.float64)


def compute_centroid_tre_um(
    fixed_pts_xyz_um: np.ndarray,
    moving_pts_xyz_um: np.ndarray,
    transform: SimilarityTransform3D,
    max_match_dist_um: float = 25.0,
) -> Tuple[float, float, float, int]:
    """Compute Target Registration Error (mean, median, p95 in um) between matched 3D landmarks."""
    f_pts = np.asarray(fixed_pts_xyz_um, dtype=np.float64)
    m_pts = np.asarray(moving_pts_xyz_um, dtype=np.float64)
    if f_pts.shape[0] == 0 or m_pts.shape[0] == 0:
        return float("nan"), float("nan"), float("nan"), 0

    mapped_f = transform.transform_points_xyz_um(f_pts)
    tree = cKDTree(m_pts)
    dists, _ = tree.query(mapped_f, k=1)
    inliers = dists[dists <= max_match_dist_um]
    if inliers.size == 0:
        return float(np.mean(dists)), float(np.median(dists)), float(np.percentile(dists, 95.0)), 0
    return (
        float(np.mean(inliers)),
        float(np.median(inliers)),
        float(np.percentile(inliers, 95.0)),
        int(inliers.size),
    )


def register_3d_similarity(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    fixed_spacing_zyx_um: Tuple[float, float, float],
    moving_spacing_zyx_um: Tuple[float, float, float],
    fixed_mask_zyx: Optional[np.ndarray] = None,
    moving_mask_zyx: Optional[np.ndarray] = None,
    fixed_origin_zyx_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    moving_origin_zyx_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    metric: str = "mattes_mi",
    histogram_bins: int = 32,
    sampling_percentage: float = 0.20,
    max_iterations: int = 200,
    shrink_factors: Sequence[int] = (4, 2, 1),
    smoothing_sigmas: Sequence[float] = (2.0, 1.0, 0.0),
    allow_scale: bool = True,
) -> LadderRegistrationResult:
    """Fit a 3D 7-DOF similarity (or 6-DOF rigid) transform using multi-resolution SimpleITK pyramid."""
    sitk = _require_sitk()
    fix_img = numpy_zyx_to_sitk(fixed_zyx, fixed_spacing_zyx_um, fixed_origin_zyx_um)
    mov_img = numpy_zyx_to_sitk(moving_zyx, moving_spacing_zyx_um, moving_origin_zyx_um)

    base_tx = sitk.Similarity3DTransform() if allow_scale else sitk.Euler3DTransform()
    init_tx = sitk.CenteredTransformInitializer(
        fix_img,
        mov_img,
        base_tx,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    if metric == "mean_squares":
        reg.SetMetricAsMeanSquares()
    else:
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(histogram_bins))

    reg.SetMetricSamplingStrategy(reg.REGULAR)
    eff_pct = float(sampling_percentage)
    active_voxels = (
        max(1, int(np.count_nonzero(fixed_mask_zyx > 0)))
        if fixed_mask_zyx is not None
        else int(fixed_zyx.size)
    )
    if active_voxels > 120_000:
        eff_pct = min(0.50, max(0.02, 120_000.0 / float(active_voxels)))
    elif fixed_mask_zyx is not None and fixed_zyx.size > 1_000_000:
        eff_pct = min(0.50, max(eff_pct, 80_000.0 / float(max(1, active_voxels)) * (active_voxels / float(fixed_zyx.size))))
    reg.SetMetricSamplingPercentage(eff_pct, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)

    if fixed_mask_zyx is not None:
        m_fix = sitk.GetImageFromArray((fixed_mask_zyx > 0).astype(np.uint8))
        m_fix.CopyInformation(fix_img)
        reg.SetMetricFixedMask(m_fix)
    if moving_mask_zyx is not None:
        m_mov = sitk.GetImageFromArray((moving_mask_zyx > 0).astype(np.uint8))
        m_mov.CopyInformation(mov_img)
        reg.SetMetricMovingMask(m_mov)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=int(max_iterations),
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Clamp shrink factors to small test volumes cleanly
    min_dim = min(min(fixed_zyx.shape), min(moving_zyx.shape))
    valid_shrinks: List[int] = []
    valid_sigmas: List[float] = []
    for sh, sg in zip(shrink_factors, smoothing_sigmas):
        if min_dim // sh >= 8:
            valid_shrinks.append(int(sh))
            valid_sigmas.append(float(sg))
    if not valid_shrinks:
        valid_shrinks = [1]
        valid_sigmas = [0.0]

    reg.SetShrinkFactorsPerLevel(shrinkFactors=valid_shrinks)
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=valid_sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetInitialTransform(init_tx, inPlace=True)

    final_tx = reg.Execute(fix_img, mov_img)
    if not allow_scale:
        if isinstance(final_tx, sitk.CompositeTransform):
            final_tx = final_tx.GetNthTransform(final_tx.GetNumberOfTransforms() - 1)
        euler = sitk.Euler3DTransform(final_tx)
        sim_wrap = sitk.Similarity3DTransform()
        sim_wrap.SetCenter(euler.GetCenter())
        sim_wrap.SetMatrix(euler.GetMatrix())
        sim_wrap.SetTranslation(euler.GetTranslation())
        sim_wrap.SetScale(1.0)
        sim_dto = SimilarityTransform3D.from_sitk(sim_wrap)
    else:
        sim_dto = SimilarityTransform3D.from_sitk(final_tx)

    f_pts = extract_feature_centroids_xyz_um(fixed_zyx, fixed_spacing_zyx_um, fixed_origin_zyx_um)
    m_pts = extract_feature_centroids_xyz_um(moving_zyx, moving_spacing_zyx_um, moving_origin_zyx_um)
    max_tol = max(fixed_spacing_zyx_um[0], moving_spacing_zyx_um[0]) * 12.0
    tre_mean, tre_med, tre_p95, n_match = compute_centroid_tre_um(f_pts, m_pts, sim_dto, max_match_dist_um=max_tol)

    return LadderRegistrationResult(
        transform=sim_dto,
        final_metric_value=float(reg.GetMetricValue()),
        stop_condition=str(reg.GetOptimizerStopConditionDescription()),
        tre_mean_um=tre_mean,
        tre_median_um=tre_med,
        tre_p95_um=tre_p95,
        num_matched_landmarks=n_match,
    )


def radial_non_cupping_mask(
    shape_zyx: Tuple[int, int, int],
    valid_radius_fraction: float = 0.85,
) -> np.ndarray:
    """Create cylindrical validity mask excluding FBP truncation cupping near the 0.325 um FOV boundary."""
    nz, ny, nx = shape_zyx
    cy, cx = 0.5 * (ny - 1), 0.5 * (nx - 1)
    r_max = float(min(cy, cx)) * float(valid_radius_fraction)
    yy, xx = np.ogrid[:ny, :nx]
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r_max * r_max)
    return np.broadcast_to(disk[None, :, :], (nz, ny, nx)).copy()


@dataclass(frozen=True)
class SphereFitResult:
    """Sub-voxel 3D calibration sphere fit on radial-normal-filtered sphere-to-air cap points."""

    center_zyx_vox: Tuple[float, float, float]
    radius_vox: float
    diameter_vox: float
    nominal_voxel_um: float
    apparent_diameter_nominal_um: float
    calibrated_voxel_um: float
    voxel_scale_error_pct: float
    sphericity_rmse_vox: float
    sphericity_rmse_um: float
    num_surface_points: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def fit_calibration_sphere_3d(
    vol_or_dataset_zyx: Any,
    nominal_voxel_um: float,
    seed_center_zyx_vox: Tuple[float, float, float],
    reference_diameter_um: float = 500.0,
    border_margin_vox: int = 12,
    smooth_sigma_vox: float = 1.5,
    max_working_radius_vox: float = 200.0,
) -> SphereFitResult:
    """Fit a 3D sphere (z - z0)^2 + (y - y0)^2 + (x - x0)^2 = R^2 to outward-normal sphere-to-air points.

    Works even when the sphere is partially clipped by a sub-volume boundary or touches a glass capillary wall
    by restricting to interior non-border points whose outward gradient normal aligns with the radial vector
    and transitions cleanly from interior sphere plateau to exterior air plateau.
    """
    nz, ny, nx = vol_or_dataset_zyx.shape
    r_est_full = (0.5 * reference_diameter_um) / float(nominal_voxel_um)
    stride = max(1, int(np.ceil(r_est_full / float(max_working_radius_vox))))
    pad = max(16 * stride, int(round(r_est_full * 1.25)))
    cz, cy, cx = [float(v) for v in seed_center_zyx_vox]
    z0, z1 = max(0, int(round(cz - pad))), min(nz, int(round(cz + pad)))
    y0, y1 = max(0, int(round(cy - pad))), min(ny, int(round(cy + pad)))
    x0, x1 = max(0, int(round(cx - pad))), min(nx, int(round(cx + pad)))

    sub = np.asarray(vol_or_dataset_zyx[z0:z1:stride, y0:y1:stride, x0:x1:stride], dtype=np.float32)
    sm = gaussian_filter(sub, sigma=float(smooth_sigma_vox))
    del sub

    eff_vox_um = float(nominal_voxel_um) * stride
    r_est = r_est_full / stride
    cz_s, cy_s, cx_s = (cz - z0) / stride, (cy - y0) / stride, (cx - x0) / stride

    half_win = max(3, min(8, min(sm.shape) // 8))
    sz = int(np.clip(cz_s, half_win + 1, sm.shape[0] - half_win - 2))
    sy = int(np.clip(cy_s, half_win + 1, sm.shape[1] - half_win - 2))
    sx = int(np.clip(cx_s, half_win + 1, sm.shape[2] - half_win - 2))
    in_val = float(
        np.median(
            sm[
                sz - half_win : sz + half_win,
                sy - half_win : sy + half_win,
                sx - half_win : sx + half_win,
            ]
        )
    )
    air_val = float(np.percentile(sm, 15.0))
    mid_val = 0.5 * (in_val + air_val)
    contrast = max(abs(in_val - air_val), 1e-4)

    gz = sobel(sm, axis=0) * np.float32(0.125)
    gy = sobel(sm, axis=1) * np.float32(0.125)
    gx = sobel(sm, axis=2) * np.float32(0.125)
    gmag = np.sqrt(gz * gz + gy * gy + gx * gx) + np.float32(1e-8)

    c_cur = np.array([cz_s, cy_s, cx_s], dtype=np.float64)
    r_cur = float(r_est)
    pts_loc_final = np.zeros((0, 3), dtype=np.float64)
    bm = max(2, int(np.ceil(border_margin_vox / stride)))

    zz_idx = np.arange(sm.shape[0], dtype=np.float32)[:, None, None]
    yy_idx = np.arange(sm.shape[1], dtype=np.float32)[None, :, None]
    xx_idx = np.arange(sm.shape[2], dtype=np.float32)[None, None, :]
    glob_z = z0 + zz_idx * stride
    glob_y = y0 + yy_idx * stride
    glob_x = x0 + xx_idx * stride
    full_ok = (
        (glob_z >= border_margin_vox)
        & (glob_z < nz - border_margin_vox)
        & (glob_y >= border_margin_vox)
        & (glob_y < ny - border_margin_vox)
        & (glob_x >= border_margin_vox)
        & (glob_x < nx - border_margin_vox)
    )

    for it in range(4):
        dz = zz_idx - np.float32(c_cur[0])
        dy = yy_idx - np.float32(c_cur[1])
        dx = xx_idx - np.float32(c_cur[2])
        rr = np.sqrt(dz * dz + dy * dy + dx * dx) + np.float32(1e-8)
        dot_rad = -(gz * dz + gy * dy + gx * dx) / (gmag * rr)
        tol_r = (0.14 if it == 0 else 0.05) * r_cur
        cand = (
            full_ok
            & (np.abs(rr - np.float32(r_cur)) < np.float32(tol_r))
            & (dot_rad > np.float32(0.96))
            & (np.abs(sm - np.float32(mid_val)) < np.float32(0.10 * contrast))
            & (gmag > np.float32(0.12 * contrast))
        )
        del dz, dy, dx, rr, dot_rad
        pts_loc = np.argwhere(cand).astype(np.float64)
        if pts_loc.shape[0] < 16:
            break
        u_vec = pts_loc - c_cur[None, :]
        u_vec /= np.linalg.norm(u_vec, axis=1, keepdims=True) + 1e-8
        step_d = max(3.0, min(0.35 * r_cur, 10.0 / eff_vox_um))
        v_in = map_coordinates(sm, (pts_loc - step_d * u_vec).T, order=1, mode="nearest")
        v_out = map_coordinates(sm, (pts_loc + step_d * u_vec).T, order=1, mode="nearest")
        v_out2 = map_coordinates(sm, (pts_loc + 1.8 * step_d * u_vec).T, order=1, mode="nearest")
        clean = (
            (np.abs(v_in - in_val) < 0.18 * contrast)
            & (np.abs(v_out - air_val) < 0.18 * contrast)
            & (np.abs(v_out2 - air_val) < 0.22 * contrast)
        )
        if np.count_nonzero(clean) >= 16:
            pts_loc = pts_loc[clean]
        if it >= 2 and pts_loc.shape[0] > 100:
            resid_prev = np.abs(np.linalg.norm(pts_loc - c_cur[None, :], axis=1) - r_cur)
            inliers = resid_prev < max(1.5, 2.5 / eff_vox_um)
            if np.count_nonzero(inliers) >= 50:
                pts_loc = pts_loc[inliers]
        if pts_loc.shape[0] > 25000:
            rng = np.random.default_rng(42 + it)
            pts_loc = pts_loc[rng.choice(pts_loc.shape[0], size=25000, replace=False)]

        pts_loc_final = pts_loc
        fit = least_squares(
            lambda p: np.linalg.norm(pts_loc_final - p[:3][None, :], axis=1) - p[3],
            np.append(c_cur, r_cur),
            loss="cauchy",
            f_scale=0.5,
        )
        c_cur, r_cur = fit.x[:3], float(fit.x[3])

    if pts_loc_final.shape[0] == 0:
        raise ValueError("No clean sphere-to-air surface points found around seed_center_zyx_vox.")

    resid_s = np.abs(np.linalg.norm(pts_loc_final - c_cur[None, :], axis=1) - r_cur)
    rmse_vox = float(np.sqrt(np.mean(resid_s * resid_s))) * stride
    c_glob = (z0 + c_cur[0] * stride, y0 + c_cur[1] * stride, x0 + c_cur[2] * stride)
    r_glob = r_cur * stride
    diam_vox = 2.0 * r_glob
    nom_diam_um = diam_vox * float(nominal_voxel_um)
    cal_vox_um = float(reference_diameter_um) / diam_vox
    err_pct = (cal_vox_um / float(nominal_voxel_um) - 1.0) * 100.0

    return SphereFitResult(
        center_zyx_vox=(float(c_glob[0]), float(c_glob[1]), float(c_glob[2])),
        radius_vox=r_glob,
        diameter_vox=diam_vox,
        nominal_voxel_um=float(nominal_voxel_um),
        apparent_diameter_nominal_um=nom_diam_um,
        calibrated_voxel_um=cal_vox_um,
        voxel_scale_error_pct=err_pct,
        sphericity_rmse_vox=rmse_vox,
        sphericity_rmse_um=rmse_vox * cal_vox_um,
        num_surface_points=int(pts_loc_final.shape[0]),
    )


def read_i13_nexus_stage_metadata(nexus_path: Path | str) -> Dict[str, Any]:
    """Read sample stage, camera bore-sight, turret/scintillator, and timing metadata from raw I13-2 .nxs."""
    fields = {
        "title": "entry1/title",
        "experiment_identifier": "entry1/experiment_identifier",
        "start_time": "entry1/start_time",
        "end_time": "entry1/end_time",
        "turret_name": "entry1/before_scan/det_cfg/turret_name",
        "focus_mm": "entry1/before_scan/det_cfg/focus",
        "translate_x_mm": "entry1/before_scan/translate_x/translate_x",
        "cs1_x_mm": "entry1/before_scan/cs1/cs1_x",
        "cs1_y_mm": "entry1/before_scan/cs1/cs1_y",
        "cs1_z_mm": "entry1/before_scan/cs1/cs1_z",
        "ss1_samplex_um": "entry1/before_scan/sample_stage/ss1_samplex",
        "ss1_sampley_um": "entry1/before_scan/sample_stage/ss1_sampley",
        "ss1_samplez_um": "entry1/before_scan/sample_stage/ss1_samplez",
        "ss1_X_mm": "entry1/before_scan/sample_stage/ss1_X",
        "ss1_Y_mm": "entry1/before_scan/sample_stage/ss1_Y",
        "ss1_Z_mm": "entry1/before_scan/sample_stage/ss1_Z",
        "ss1_pitch_deg": "entry1/before_scan/sample_stage/ss1_pitch",
        "ss1_roll_deg": "entry1/before_scan/sample_stage/ss1_roll",
        "detector_distance_mm": "entry1/tomo_entry/instrument/detector/distance",
        "nominal_pixel_size_um": "entry1/tomo_entry/instrument/detector/x_pixel_size",
    }
    out: Dict[str, Any] = {}
    with h5py.File(nexus_path, "r") as fh:
        for key, h5_path in fields.items():
            if h5_path not in fh:
                continue
            val = fh[h5_path][()]
            if isinstance(val, bytes):
                sval = val.decode("utf-8", "replace")
                try:
                    out[key] = float(sval) if key.endswith("_mm") else sval
                except ValueError:
                    out[key] = sval
            elif isinstance(val, np.generic):
                out[key] = val.item()
            else:
                out[key] = val
    return out

