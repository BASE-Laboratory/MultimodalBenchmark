"""MultimodalBenchmark: 3D synchrotron XCT + XRD-CT benchmark suite."""

__version__ = "2.0.1"

from .manifest import DatasetManifest, load_manifest, resolve_doi, verify_file_md5
from .io import (
    load_ascii_xrd,
    load_xct_subvolume,
    load_xrdct_sinogram,
    load_xrdct_peak_maps,
    load_xrdct_volume_50um,
    two_theta_to_q,
)

__all__ = [
    "__version__",
    "DatasetManifest",
    "load_manifest",
    "resolve_doi",
    "verify_file_md5",
    "load_ascii_xrd",
    "load_xct_subvolume",
    "load_xrdct_sinogram",
    "load_xrdct_peak_maps",
    "load_xrdct_volume_50um",
    "two_theta_to_q",
]
