"""Drop-in replacement adapter for BASE-Laboratory/X-FUSE (`x_fuse.loaders`).

Fixes five upstream issues discovered during the MultimodalBenchmark v2.0 audit:
1. Out-of-core HDF5 slicing on the 57 GB DIAD Astra XCT volume (avoids loading full 57 GB before cropping).
2. Explicit scan-ID selection for 50 um vs 25 um XRD-CT files (replaces fragile `crop=slice(5, -1)` directory slice).
3. Tolerance-matched / multi-reflection aggregation across all 13-15 Bragg peaks (fixes `q~1.655` vs `q~1.656` KeyError
   and reduces 0.0049 A^-1 Na/Zn doublet crosstalk).
4. Corrects the 47.5 deg rotation offset in pre-reconstructed 50 um slices caused by the `gts_cs_theta = [-47.5, +312.5]`
   soft-motor zero offset relative to the `[0, 360]` deg 25 um ROI scans.
5. Enforces the +/-100 um axial z-guard band around held-out 25 um ROI test slices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from multimodal_benchmark.io import (
    aggregate_peak_maps,
    load_xct_subvolume,
    load_xrdct_peak_maps,
    load_xrdct_volume_50um,
)

DEFAULT_SPLITS_PATH = Path(__file__).resolve().parents[2] / "splits" / "benchmark_splits_v1.json"


def load_xfuse_benchmark_t3_bundle(
    xrdct_dir: Path | str,
    diad_xct_h5: Optional[Path | str] = None,
    splits_path: Path | str = DEFAULT_SPLITS_PATH,
    aggregation: str = "all_sum",
    flip_xrd_x: bool = True,
    xct_zyx_crop: Tuple[slice, slice, slice] = (slice(150, 2000), slice(350, 2200), slice(350, 2200)),
    xct_stride: int = 1,
) -> Dict[str, Any]:
    """Load guard-banded 50 um training guides, held-out 25 um test slices, and optional out-of-core DIAD XCT."""
    xdir = Path(xrdct_dir)
    with open(splits_path, "r", encoding="utf-8") as fh:
        t3_splits = json.load(fh)["t3_correlative_fusion"]

    allowed_50um: Sequence[int] = t3_splits["train_self_supervised_50um_scans"]
    test_25um: Sequence[int] = t3_splits["held_out_test_25um_scans"]

    na_50_all, all_50_ids = load_xrdct_volume_50um(xdir, phase="na", aggregation=aggregation, flip_x=flip_xrd_x)
    zn_50_all, _ = load_xrdct_volume_50um(xdir, phase="zn", aggregation=aggregation, flip_x=flip_xrd_x)

    train_mask_50 = np.array([sid in set(allowed_50um) for sid in all_50_ids], dtype=bool)

    test_slices_25: Dict[int, Dict[str, np.ndarray]] = {}
    for sid in test_25um:
        f_na = xdir / f"output_{sid}_na.nxs"
        f_zn = xdir / f"output_{sid}_zn.nxs"
        if f_na.exists() and f_zn.exists():
            p_na = load_xrdct_peak_maps(f_na, flip_x=flip_xrd_x)
            p_zn = load_xrdct_peak_maps(f_zn, flip_x=flip_xrd_x)
            test_slices_25[int(sid)] = {
                "na": aggregate_peak_maps(p_na, mode=aggregation, target_q=1.651),
                "zn": aggregate_peak_maps(p_zn, mode=aggregation, target_q=1.656),
            }

    out: Dict[str, Any] = {
        "guide_50um_zyx": {"Na": na_50_all, "Zn": zn_50_all},
        "scan_ids_50um": all_50_ids,
        "train_self_sup_mask_50um": train_mask_50,
        "held_out_test_25um": test_slices_25,
    }

    if diad_xct_h5 is not None and Path(diad_xct_h5).exists():
        out["diad_xct_zyx"] = load_xct_subvolume(
            diad_xct_h5,
            zyx_slices=xct_zyx_crop,
            stride=xct_stride,
            canonical_zyx=True,
        )
    return out
