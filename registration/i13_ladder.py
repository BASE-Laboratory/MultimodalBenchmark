"""Stage 0B: I13-2 multi-magnification 7-DOF similarity registration, landmark TRE, and local-tomography mask."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass, label as nd_label
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu


@dataclass(frozen=True)
class SimilarityTransform3D:
    """3D 7-DOF similarity transform in physical micron coordinates (SimpleITK XYZ convention)."""

    matrix_3x3: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    translation_xyz_um: Tuple[float, float, float]
    center_xyz_um: Tuple[float, float, float]
    scale: float
    versor_xyzw: Tuple[float, float, float, float]

    def transform_points_xyz_um(self, pts_xyz_um: np.ndarray) -> np.ndarray:
        """Map fixed physical points (N, 3) in (x, y, z) um to moving physical points."""
        pts = np.asarray(pts_xyz_um, dtype=np.float64)
        mat = np.asarray(self.matrix_3x3, dtype=np.float64)
        cen = np.asarray(self.center_xyz_um, dtype=np.float64)
        trn = np.asarray(self.translation_xyz_um, dtype=np.float64)
        return (pts - cen) @ mat.T + cen + trn

    def to_sitk(self) -> sitk.Similarity3DTransform:
        tx = sitk.Similarity3DTransform()
        tx.SetCenter(self.center_xyz_um)
        tx.SetRotation(self.versor_xyzw)
        tx.SetTranslation(self.translation_xyz_um)
        tx.SetScale(self.scale)
        return tx

    @classmethod
    def from_sitk(cls, tx: sitk.Transform) -> "SimilarityTransform3D":
        if isinstance(tx, sitk.CompositeTransform):
            tx = tx.GetNthTransform(tx.GetNumberOfTransforms() - 1)
        sim = sitk.Similarity3DTransform(tx)
        mat = np.asarray(sim.GetMatrix(), dtype=np.float64).reshape(3, 3)
        return cls(
            matrix_3x3=tuple(tuple(float(v) for v in row) for row in mat),
            translation_xyz_um=tuple(float(v) for v in sim.GetTranslation()),
            center_xyz_um=tuple(float(v) for v in sim.GetCenter()),
            scale=float(sim.GetScale()),
            versor_xyzw=tuple(float(v) for v in sim.GetVersor()),
        )

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
) -> sitk.Image:
    """Convert a (Z, Y, X) NumPy volume to a SimpleITK Image with physical (X, Y, Z) spacing in um."""
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
    transform: SimilarityTransform3D,
    moving_origin_zyx_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_origin_zyx_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    interpolator: int = sitk.sitkLinear,
) -> np.ndarray:
    """Resample moving 3D volume onto fixed physical grid using a 7-DOF SimilarityTransform3D."""
    mov_img = numpy_zyx_to_sitk(moving_zyx, moving_spacing_zyx_um, moving_origin_zyx_um)
    fz, fy, fx = fixed_shape_zyx
    sz, sy, sx = fixed_spacing_zyx_um
    oz, oy, ox = fixed_origin_zyx_um

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(fx), int(fy), int(fz)))
    resampler.SetOutputSpacing((float(sx), float(sy), float(sz)))
    resampler.SetOutputOrigin((float(ox), float(oy), float(oz)))
    resampler.SetTransform(transform.to_sitk())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(np.percentile(moving_zyx, 5.0)))
    out_img = resampler.Execute(mov_img)
    return sitk.GetArrayFromImage(out_img).astype(np.float32)


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
    if fixed_zyx.size > 1_000_000:
        eff_pct = min(eff_pct, max(0.015, 120_000.0 / float(fixed_zyx.size)))
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
