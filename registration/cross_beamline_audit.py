"""Stage 0C: Cross-beamline (I13-2 <-> DIAD) fiducial-vs-pellet rigid slip & octant deformation audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.ndimage import binary_erosion

from .i13_ladder import (
    LadderRegistrationResult,
    SimilarityTransform3D,
    register_3d_similarity,
    resample_moving_to_fixed_zyx,
)


@dataclass(frozen=True)
class OctantDriftEntry:
    octant_zyx_index: Tuple[int, int, int]
    bbox_zyx: Tuple[int, int, int, int, int, int]
    solid_voxels: int
    local_vs_global_disp_um: float
    local_scale: float
    certified: bool


@dataclass(frozen=True)
class Stage0CAuditReport:
    status: str
    fiducial_transform: SimilarityTransform3D
    pellet_transform: SimilarityTransform3D
    pellet_vs_fiducial_slip_um: float
    pellet_moved_inside_capillary: bool
    max_octant_drift_um: float
    mean_octant_drift_um: float
    octant_results: List[OctantDriftEntry]
    certified_octant_count: int
    phase_cnr_zn_vs_na: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_certified_regions_yaml(self, path: Path | str) -> None:
        payload = {
            "stage0c_status": self.status,
            "pellet_vs_fiducial_slip_um": round(self.pellet_vs_fiducial_slip_um, 4),
            "pellet_moved_inside_capillary": self.pellet_moved_inside_capillary,
            "max_octant_drift_um": round(self.max_octant_drift_um, 4),
            "phase_cnr_zn_vs_na": round(self.phase_cnr_zn_vs_na, 4) if self.phase_cnr_zn_vs_na is not None else None,
            "t2_cross_beamline_sr_enabled": self.status in (
                "GLOBAL_SIMILARITY_CERTIFIED",
                "PIECEWISE_OCTANT_CERTIFIED",
            ),
            "t4b_direct_xct_phase_seg_feasible": (
                bool(self.phase_cnr_zn_vs_na >= 1.5) if self.phase_cnr_zn_vs_na is not None else None
            ),
            "global_pellet_transform": asdict(self.pellet_transform),
            "fiducial_transform": asdict(self.fiducial_transform),
            "certified_octants": [asdict(o) for o in self.octant_results if o.certified],
            "rejected_octants": [asdict(o) for o in self.octant_results if not o.certified],
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)


def match_histograms_masked(
    source: np.ndarray,
    reference: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    num_quantiles: int = 256,
) -> np.ndarray:
    """Match empirical intensity quantiles of source to reference inside foreground masks."""
    src = np.asarray(source, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    s_vals = src[source_mask > 0] if source_mask is not None and np.any(source_mask > 0) else src.ravel()
    r_vals = ref[reference_mask > 0] if reference_mask is not None and np.any(reference_mask > 0) else ref.ravel()

    qs = np.linspace(0.0, 100.0, num_quantiles)
    s_quant = np.percentile(s_vals, qs)
    r_quant = np.percentile(r_vals, qs)
    # Ensure strict monotonicity for np.interp
    s_unique, idx = np.unique(s_quant, return_index=True)
    r_unique = r_quant[idx]
    if s_unique.size < 2:
        return src.copy()
    matched = np.interp(src.ravel(), s_unique, r_unique).reshape(src.shape)
    return matched.astype(np.float32)


def compute_phase_cnr(
    xct_zyx: np.ndarray,
    zn_mask_zyx: np.ndarray,
    na_mask_zyx: np.ndarray,
    erosion_iters: int = 1,
) -> float:
    """Compute Contrast-to-Noise Ratio CNR(Zn-13X, Na-13X) inside eroded solid interiors (away from phase-contrast fringes)."""
    vol = np.asarray(xct_zyx, dtype=np.float64)
    zm = zn_mask_zyx > 0
    nm = na_mask_zyx > 0
    if erosion_iters > 0:
        zm_er = binary_erosion(zm, iterations=erosion_iters)
        nm_er = binary_erosion(nm, iterations=erosion_iters)
        if np.any(zm_er):
            zm = zm_er
        if np.any(nm_er):
            nm = nm_er
    if not (np.any(zm) and np.any(nm)):
        return 0.0
    zn_vals = vol[zm]
    na_vals = vol[nm]
    denom = np.sqrt(np.var(zn_vals) + np.var(na_vals)) + 1e-12
    return float(abs(np.mean(zn_vals) - np.mean(na_vals)) / denom)


def _mean_transform_displacement_um(
    tx_a: SimilarityTransform3D,
    tx_b: SimilarityTransform3D,
    pts_xyz_um: np.ndarray,
) -> float:
    pa = tx_a.transform_points_xyz_um(pts_xyz_um)
    pb = tx_b.transform_points_xyz_um(pts_xyz_um)
    return float(np.mean(np.linalg.norm(pa - pb, axis=1)))


def save_checkerboard_qc_png(
    fixed_slice: np.ndarray,
    aligned_moving_slice: np.ndarray,
    out_path: Path | str,
    tile_px: int = 16,
) -> None:
    """Render a 2D checkerboard & difference QC overlay between fixed DIAD and aligned I13-2 slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _norm01(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(arr, [2.0, 98.0])
        return np.clip((arr - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    f01 = _norm01(fixed_slice)
    m01 = _norm01(aligned_moving_slice)
    ny, nx = f01.shape
    yy, xx = np.indices((ny, nx))
    grid = ((yy // tile_px) + (xx // tile_px)) % 2 == 0
    checker = np.where(grid, f01, m01)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    axes[0].imshow(f01, cmap="gray")
    axes[0].set_title("Fixed (DIAD XCT)")
    axes[1].imshow(checker, cmap="gray")
    axes[1].set_title("Checkerboard Overlay")
    im = axes[2].imshow(f01 - m01, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Signed Residual")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_stage0c_pellet_vs_fiducial_audit(
    diad_zyx: np.ndarray,
    i13_zyx: np.ndarray,
    diad_spacing_zyx_um: Tuple[float, float, float],
    i13_spacing_zyx_um: Tuple[float, float, float],
    diad_fiducial_mask_zyx: np.ndarray,
    diad_pellet_mask_zyx: np.ndarray,
    slip_threshold_um: float = 1.5,
    octant_cert_threshold_um: float = 1.0,
    min_octant_voxels: int = 64,
    zn_mask_zyx: Optional[np.ndarray] = None,
    na_mask_zyx: Optional[np.ndarray] = None,
    qc_png_path: Optional[Path | str] = None,
) -> Stage0CAuditReport:
    """Execute the Stage 0C cross-beamline diagnostic protocol:

    1. Histogram-match moving I13-2 volume to fixed DIAD volume.
    2. Fit 7-DOF similarity transform on fiducials (capillary wall + ruby/NIST spheres).
    3. Fit independent 7-DOF similarity transform on masked Zn-13X pellet.
    4. Compare pellet vs fiducial transforms at pellet coordinates to detect intra-capillary slip.
    5. Partition pellet into 2x2x2 octants and fit local rigid refinements to detect non-rigid swelling/settling.
    """
    i13_matched = match_histograms_masked(i13_zyx, diad_zyx)

    fid_res: LadderRegistrationResult = register_3d_similarity(
        fixed_zyx=diad_zyx,
        moving_zyx=i13_matched,
        fixed_spacing_zyx_um=diad_spacing_zyx_um,
        moving_spacing_zyx_um=i13_spacing_zyx_um,
        fixed_mask_zyx=diad_fiducial_mask_zyx,
        metric="mattes_mi",
        allow_scale=True,
    )

    pel_res: LadderRegistrationResult = register_3d_similarity(
        fixed_zyx=diad_zyx,
        moving_zyx=i13_matched,
        fixed_spacing_zyx_um=diad_spacing_zyx_um,
        moving_spacing_zyx_um=i13_spacing_zyx_um,
        fixed_mask_zyx=diad_pellet_mask_zyx,
        metric="mattes_mi",
        allow_scale=True,
    )

    # Sample physical coordinates inside pellet mask to measure fiducial-vs-pellet displacement
    zz, yy, xx = np.where(diad_pellet_mask_zyx > 0)
    if zz.size == 0:
        raise ValueError("diad_pellet_mask_zyx is empty; cannot audit pellet registration.")
    step = max(1, zz.size // 2000)
    sz, sy, sx = diad_spacing_zyx_um
    pel_pts_xyz_um = np.column_stack([xx[::step] * sx, yy[::step] * sy, zz[::step] * sz])

    slip_um = _mean_transform_displacement_um(fid_res.transform, pel_res.transform, pel_pts_xyz_um)
    pellet_slipped = bool(slip_um > slip_threshold_um)

    # Resample I13-2 using global pellet transform first so octant residuals directly measure local drift!
    i13_in_diad_pellet_frame = resample_moving_to_fixed_zyx(
        moving_zyx=i13_matched,
        moving_spacing_zyx_um=i13_spacing_zyx_um,
        fixed_shape_zyx=diad_zyx.shape,
        fixed_spacing_zyx_um=diad_spacing_zyx_um,
        transform=pel_res.transform,
    )

    if qc_png_path is not None:
        mid_z = diad_zyx.shape[0] // 2
        save_checkerboard_qc_png(diad_zyx[mid_z], i13_in_diad_pellet_frame[mid_z], qc_png_path)

    nz, ny, nx = diad_zyx.shape
    z_splits = [0, nz // 2, nz]
    y_splits = [0, ny // 2, ny]
    x_splits = [0, nx // 2, nx]

    octants: List[OctantDriftEntry] = []
    for iz in range(2):
        for iy in range(2):
            for ix in range(2):
                z0, z1 = z_splits[iz], z_splits[iz + 1]
                y0, y1 = y_splits[iy], y_splits[iy + 1]
                x0, x1 = x_splits[ix], x_splits[ix + 1]
                sub_mask = diad_pellet_mask_zyx[z0:z1, y0:y1, x0:x1]
                n_vox = int(np.count_nonzero(sub_mask > 0))
                if n_vox < min_octant_voxels:
                    continue

                sub_fix = diad_zyx[z0:z1, y0:y1, x0:x1]
                sub_mov = i13_in_diad_pellet_frame[z0:z1, y0:y1, x0:x1]
                origin_um = (z0 * sz, y0 * sy, x0 * sx)

                local_res = register_3d_similarity(
                    fixed_zyx=sub_fix,
                    moving_zyx=sub_mov,
                    fixed_spacing_zyx_um=diad_spacing_zyx_um,
                    moving_spacing_zyx_um=diad_spacing_zyx_um,
                    fixed_mask_zyx=sub_mask,
                    fixed_origin_zyx_um=origin_um,
                    moving_origin_zyx_um=origin_um,
                    metric="mattes_mi",
                    max_iterations=80,
                    shrink_factors=(2, 1),
                    smoothing_sigmas=(1.0, 0.0),
                    allow_scale=False,
                )

                ozz, oyy, oxx = np.where(sub_mask > 0)
                ostep = max(1, ozz.size // 500)
                opts_xyz = np.column_stack(
                    [(x0 + oxx[::ostep]) * sx, (y0 + oyy[::ostep]) * sy, (z0 + ozz[::ostep]) * sz]
                )
                mapped_opts = local_res.transform.transform_points_xyz_um(opts_xyz)
                drift_um = float(np.mean(np.linalg.norm(mapped_opts - opts_xyz, axis=1)))
                octants.append(
                    OctantDriftEntry(
                        octant_zyx_index=(iz, iy, ix),
                        bbox_zyx=(z0, z1, y0, y1, x0, x1),
                        solid_voxels=n_vox,
                        local_vs_global_disp_um=drift_um,
                        local_scale=local_res.transform.scale,
                        certified=bool(drift_um <= octant_cert_threshold_um),
                    )
                )

    drifts = [o.local_vs_global_disp_um for o in octants] or [0.0]
    max_drift = float(np.max(drifts))
    mean_drift = float(np.mean(drifts))
    n_cert = sum(1 for o in octants if o.certified)

    if max_drift <= octant_cert_threshold_um and len(octants) > 0:
        status = "GLOBAL_SIMILARITY_CERTIFIED"
    elif n_cert > 0:
        status = "PIECEWISE_OCTANT_CERTIFIED"
    else:
        status = "T2_DEGRADED_OR_RESTRICTED"

    cnr_val: Optional[float] = None
    if zn_mask_zyx is not None and na_mask_zyx is not None:
        cnr_val = compute_phase_cnr(diad_zyx, zn_mask_zyx, na_mask_zyx)

    return Stage0CAuditReport(
        status=status,
        fiducial_transform=fid_res.transform,
        pellet_transform=pel_res.transform,
        pellet_vs_fiducial_slip_um=slip_um,
        pellet_moved_inside_capillary=pellet_slipped,
        max_octant_drift_um=max_drift,
        mean_octant_drift_um=mean_drift,
        octant_results=octants,
        certified_octant_count=n_cert,
        phase_cnr_zn_vs_na=cnr_val,
    )
