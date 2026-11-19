"""Conservative, edge-preserving artifact cleanup and spatial masking utilities for synchrotron XCT & XRD-CT."""

from .artifact_cleanup import (
    ArtifactCleanupReport,
    compute_ring_anova_r2,
    compute_ruby_beam_hardening_clean_z_mask,
    detect_ring_center_yx,
    detrend_local_tomo_cupping_3d,
    interpolate_missing_xrdct_projection,
    remove_rings_polar_conservative,
    segment_pores_fringe_guarded,
)

__all__ = [
    "ArtifactCleanupReport",
    "compute_ring_anova_r2",
    "compute_ruby_beam_hardening_clean_z_mask",
    "detect_ring_center_yx",
    "detrend_local_tomo_cupping_3d",
    "interpolate_missing_xrdct_projection",
    "remove_rings_polar_conservative",
    "segment_pores_fringe_guarded",
]
