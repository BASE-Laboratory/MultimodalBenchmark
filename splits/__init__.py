"""Leakage-free 3D spatial & axial guard-banded split generator."""

from .generate_splits import (
    BoundingBox3D,
    generate_benchmark_splits,
    verify_no_spatial_leakage,
    verify_subvolume_percolation,
)

__all__ = [
    "BoundingBox3D",
    "generate_benchmark_splits",
    "verify_no_spatial_leakage",
    "verify_subvolume_percolation",
]
