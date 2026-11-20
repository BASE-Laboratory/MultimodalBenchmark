"""Split and guard-band data bundle adapter for BASE-Laboratory/X-FUSE (`x_fuse` v1.0.0+).

Note on upstream parity:
  `X-FUSE` v1.0.0 (`x_fuse.loaders`) already implements out-of-core Savu HDF5 slicing, explicit
  DIAD scan-ID selection, multi-reflection peak aggregation, and the `+47.5 deg` 50 um motor-zero
  rotation correction upstream. This benchmark adapter strictly supplies Zn-13X-Bench's 4-fold
  Leave-One-Slice-Out (LOSO) split masks (`t2_chemical_fusion_loso`) and `+/-100 um` axial guard
  bands so `xfuse fuse` / `xfuse fuse-3d` runs evaluate on identical folds as external baselines.
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


def load_xfuse_benchmark_t2_bundle(
    xrdct_dir: Path | str,
    diad_xct_h5: Optional[Path | str] = None,
    splits_path: Path | str = DEFAULT_SPLITS_PATH,
    loso_fold: Optional[int] = None,
    aggregation: str = "all_sum",
    flip_xrd_x: bool = True,
    xct_zyx_crop: Tuple[slice, slice, slice] = (slice(150, 2000), slice(350, 2200), slice(350, 2200)),
    xct_stride: int = 1,
) -> Dict[str, Any]:
    """Load guard-banded 50 um guides, 4-fold LOSO train/test 25 um slices, and optional out-of-core DIAD XCT."""
    xdir = Path(xrdct_dir)
    with open(splits_path, "r", encoding="utf-8") as fh:
        raw_splits = json.load(fh)
    t2_splits = raw_splits.get("t2_chemical_fusion_loso", raw_splits.get("t3_correlative_fusion"))
    if t2_splits is None:
        raise KeyError(f"Expected 't2_chemical_fusion_loso' in {splits_path}")

    fold_meta: Optional[Dict[str, Any]] = None
    if loso_fold is not None:
        folds = list(t2_splits["loso_folds"].values())
        matching = [f for f in folds if int(f["fold_index"]) == int(loso_fold)]
        if not matching:
            raise ValueError(f"Unknown LOSO fold index {loso_fold} (valid: 0..{len(folds) - 1})")
        fold_meta = matching[0]
        allowed_50um: Sequence[int] = fold_meta["allowed_self_supervised_50um_scans"]
        test_25um: Sequence[int] = [int(fold_meta["test_25um_scan"])]
        train_25um: Sequence[int] = [int(s) for s in fold_meta["train_25um_scans"]]
    else:
        allowed_50um = t2_splits["train_self_supervised_50um_scans"]
        test_25um = t2_splits["held_out_test_25um_scans"]
        train_25um = []

    na_50_all, all_50_ids = load_xrdct_volume_50um(xdir, phase="na", aggregation=aggregation, flip_x=flip_xrd_x)
    zn_50_all, _ = load_xrdct_volume_50um(xdir, phase="zn", aggregation=aggregation, flip_x=flip_xrd_x)
    train_mask_50 = np.array([sid in set(allowed_50um) for sid in all_50_ids], dtype=bool)

    def _load_25_dict(scan_list: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
        res: Dict[int, Dict[str, np.ndarray]] = {}
        for sid in scan_list:
            f_na = xdir / f"output_{sid}_na.nxs"
            f_zn = xdir / f"output_{sid}_zn.nxs"
            if f_na.exists() and f_zn.exists():
                p_na = load_xrdct_peak_maps(f_na, flip_x=flip_xrd_x)
                p_zn = load_xrdct_peak_maps(f_zn, flip_x=flip_xrd_x)
                res[int(sid)] = {
                    "na": aggregate_peak_maps(p_na, mode=aggregation, target_q=1.651),
                    "zn": aggregate_peak_maps(p_zn, mode=aggregation, target_q=1.656),
                }
        return res

    out: Dict[str, Any] = {
        "loso_fold": loso_fold,
        "fold_metadata": fold_meta,
        "guide_50um_zyx": {"Na": na_50_all, "Zn": zn_50_all},
        "scan_ids_50um": all_50_ids,
        "train_self_sup_mask_50um": train_mask_50,
        "train_25um_slices": _load_25_dict(train_25um),
        "held_out_test_25um": _load_25_dict(test_25um),
    }

    if diad_xct_h5 is not None and Path(diad_xct_h5).exists():
        out["diad_xct_zyx"] = load_xct_subvolume(
            diad_xct_h5,
            zyx_slices=xct_zyx_crop,
            stride=xct_stride,
            canonical_zyx=True,
        )
    return out


# Backward-compatible alias
load_xfuse_benchmark_t3_bundle = load_xfuse_benchmark_t2_bundle
