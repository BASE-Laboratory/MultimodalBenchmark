"""Task T3 cross-beamline registration evaluation metrics (I13-2 <-> DIAD XCT, gated on Stage 0C).

Scores candidate 3-DOF rigid, 7-DOF similarity, or deformable (e.g. VoxelMorph) transforms against the
Stage 0C reference transform using:
1. Disjoint fiducial (or held-out landmark) Target Registration Error (TRE mean, median, P95 in um).
2. Isotropic magnification / voxel-scale recovery error (%).
3. Local 2x2x2 octant residual drift (um) inside the catalyst pellet.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def apply_affine_or_displacement_xyz(
    points_xyz_um: np.ndarray,
    matrix_3x3: Optional[np.ndarray] = None,
    translation_xyz_um: Optional[np.ndarray] = None,
    center_xyz_um: Optional[np.ndarray] = None,
    mapped_points_xyz_um: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map 3D physical coordinates (N, 3) in um via either an explicit affine/7-DOF matrix or pre-warped points."""
    if mapped_points_xyz_um is not None:
        out = np.asarray(mapped_points_xyz_um, dtype=np.float64)
        if out.shape != points_xyz_um.shape:
            raise ValueError(f"Expected mapped_points_xyz_um shape {points_xyz_um.shape}, got {out.shape}")
        return out
    pts = np.asarray(points_xyz_um, dtype=np.float64)
    mat = np.eye(3, dtype=np.float64) if matrix_3x3 is None else np.asarray(matrix_3x3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64) if translation_xyz_um is None else np.asarray(translation_xyz_um, dtype=np.float64)
    c = np.zeros(3, dtype=np.float64) if center_xyz_um is None else np.asarray(center_xyz_um, dtype=np.float64)
    return (pts - c[None, :]) @ mat.T + c[None, :] + t[None, :]


def estimate_effective_scale_from_jacobian(
    points_xyz_um: np.ndarray,
    mapped_points_xyz_um: np.ndarray,
) -> float:
    """Estimate global isotropic scale factor from RMS pairwise distance ratio around centroid."""
    p0 = np.asarray(points_xyz_um, dtype=np.float64)
    p1 = np.asarray(mapped_points_xyz_um, dtype=np.float64)
    r0 = np.linalg.norm(p0 - np.mean(p0, axis=0, keepdims=True), axis=1)
    r1 = np.linalg.norm(p1 - np.mean(p1, axis=0, keepdims=True), axis=1)
    denom = float(np.sqrt(np.mean(r0 * r0))) + 1e-12
    return float(np.sqrt(np.mean(r1 * r1)) / denom)


def compute_t3_registration_metrics(
    fiducial_points_xyz_um: np.ndarray,
    ref_mapped_fiducials_xyz_um: np.ndarray,
    pred_mapped_fiducials_xyz_um: np.ndarray,
    ref_scale: Optional[float] = None,
    pred_scale: Optional[float] = None,
    pellet_octant_centers_xyz_um: Optional[np.ndarray] = None,
    ref_mapped_octants_xyz_um: Optional[np.ndarray] = None,
    pred_mapped_octants_xyz_um: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Task T3 cross-beamline registration metrics (fiducial TRE, scale error, and octant residual drift)."""
    f_src = np.asarray(fiducial_points_xyz_um, dtype=np.float64)
    f_ref = np.asarray(ref_mapped_fiducials_xyz_um, dtype=np.float64)
    f_prd = np.asarray(pred_mapped_fiducials_xyz_um, dtype=np.float64)
    if f_ref.shape != f_prd.shape or f_ref.ndim != 2 or f_ref.shape[1] != 3:
        raise ValueError(f"Invalid fiducial point array shapes: pred {f_prd.shape} vs ref {f_ref.shape}")

    tre_vec = np.linalg.norm(f_prd - f_ref, axis=1)
    s_ref = float(ref_scale) if ref_scale is not None else estimate_effective_scale_from_jacobian(f_src, f_ref)
    s_prd = float(pred_scale) if pred_scale is not None else estimate_effective_scale_from_jacobian(f_src, f_prd)
    scale_rel_err_pct = 100.0 * abs(s_prd - s_ref) / max(abs(s_ref), 1e-12)

    out: Dict[str, float] = {
        "fiducial_tre_mean_um": float(np.mean(tre_vec)),
        "fiducial_tre_median_um": float(np.median(tre_vec)),
        "fiducial_tre_p95_um": float(np.percentile(tre_vec, 95.0)),
        "fiducial_tre_max_um": float(np.max(tre_vec)),
        "scale_pred": float(s_prd),
        "scale_ref": float(s_ref),
        "scale_rel_err_pct": float(scale_rel_err_pct),
    }

    if (
        pellet_octant_centers_xyz_um is not None
        and ref_mapped_octants_xyz_um is not None
        and pred_mapped_octants_xyz_um is not None
    ):
        o_ref = np.asarray(ref_mapped_octants_xyz_um, dtype=np.float64)
        o_prd = np.asarray(pred_mapped_octants_xyz_um, dtype=np.float64)
        err_oct = o_prd - o_ref
        # Residual internal octant drift after removing global mean translation bias
        drift_oct = np.linalg.norm(err_oct - np.mean(err_oct, axis=0, keepdims=True), axis=1)
        out["octant_residual_drift_mean_um"] = float(np.mean(drift_oct))
        out["octant_residual_drift_max_um"] = float(np.max(drift_oct))
        out["pellet_tre_mean_um"] = float(np.mean(np.linalg.norm(err_oct, axis=1)))
    else:
        err_f = f_prd - f_ref
        drift_f = np.linalg.norm(err_f - np.mean(err_f, axis=0, keepdims=True), axis=1)
        out["octant_residual_drift_mean_um"] = float(np.mean(drift_f))
        out["octant_residual_drift_max_um"] = float(np.max(drift_f))

    return out
