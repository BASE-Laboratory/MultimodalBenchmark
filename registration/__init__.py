"""Stage 0 audit, registration, and reference-noise estimation suite."""

from .xrdct_reference import (
    CORResult,
    FRCResult,
    estimate_conjugate_cor,
    reconstruct_sinogram_2d,
    unmix_sinogram_nnls,
    compute_2d_frc,
    estimate_odd_even_noise_floor,
    estimate_reflection_subset_noise_floor,
    audit_cached_xrdct_outputs,
)
from .diad_correlative import (
    CorrelativeAlignmentResult,
    blur_xct_to_xrdct_grid,
    align_xrdct_slice_to_diad_xct,
)
from .i13_ladder import (
    SimilarityTransform3D,
    LadderRegistrationResult,
    register_3d_similarity,
    compute_centroid_tre_um,
    radial_non_cupping_mask,
)
from .cross_beamline_audit import (
    Stage0CAuditReport,
    compute_phase_cnr,
    run_stage0c_pellet_vs_fiducial_audit,
)

__all__ = [
    "CORResult",
    "FRCResult",
    "estimate_conjugate_cor",
    "reconstruct_sinogram_2d",
    "unmix_sinogram_nnls",
    "compute_2d_frc",
    "estimate_odd_even_noise_floor",
    "estimate_reflection_subset_noise_floor",
    "audit_cached_xrdct_outputs",
    "CorrelativeAlignmentResult",
    "blur_xct_to_xrdct_grid",
    "align_xrdct_slice_to_diad_xct",
    "SimilarityTransform3D",
    "LadderRegistrationResult",
    "register_3d_similarity",
    "compute_centroid_tre_um",
    "radial_non_cupping_mask",
    "Stage0CAuditReport",
    "compute_phase_cnr",
    "run_stage0c_pellet_vs_fiducial_audit",
]
