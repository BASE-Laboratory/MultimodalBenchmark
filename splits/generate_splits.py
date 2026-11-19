"""Leakage-free 3D sub-volume split generator with physical spatial & axial guard bands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundingBox3D:
    split: str
    block_id: str
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int
    voxel_size_um: float

    @property
    def bounds_um(self) -> Tuple[float, float, float, float, float, float]:
        v = self.voxel_size_um
        return (self.z0 * v, self.z1 * v, self.y0 * v, self.y1 * v, self.x0 * v, self.x1 * v)


def min_separation_um(a: BoundingBox3D, b: BoundingBox3D) -> float:
    """Compute exact Euclidean distance (in um) between two 3D axis-aligned bounding boxes."""
    az0, az1, ay0, ay1, ax0, ax1 = a.bounds_um
    bz0, bz1, by0, by1, bx0, bx1 = b.bounds_um
    dz = max(0.0, az0 - bz1, bz0 - az1)
    dy = max(0.0, ay0 - by1, by0 - ay1)
    dx = max(0.0, ax0 - bx1, bx0 - ax1)
    return float(np.sqrt(dz * dz + dy * dy + dx * dx))


def verify_no_spatial_leakage(
    boxes: Sequence[BoundingBox3D],
    min_guard_band_um: float = 32.0,
) -> Tuple[bool, float]:
    """Verify that every pair of boxes belonging to different splits is separated by >= min_guard_band_um."""
    worst_sep = float("inf")
    for i, ba in enumerate(boxes):
        for bb in boxes[i + 1 :]:
            if ba.split == bb.split:
                continue
            sep = min_separation_um(ba, bb)
            worst_sep = min(worst_sep, sep)
            if sep + 1e-6 < min_guard_band_um:
                return False, sep
    return True, worst_sep


def verify_subvolume_percolation(binary_pore_zyx: np.ndarray, min_porosity: float = 0.05) -> Dict[str, Any]:
    """Verify via OpenImpala that a candidate test sub-volume has sufficient porosity and percolates across all 3 axes."""
    import openimpala as oi

    arr = (np.asarray(binary_pore_zyx) > 0).astype(np.uint8)
    with oi.Session():
        eps = float(oi.volume_fraction(arr, phase=1).fraction)
        axes_ok = {ax: bool(oi.percolation_check(arr, phase=1, direction=ax).percolates) for ax in ("x", "y", "z")}
    certified = bool(eps >= min_porosity and all(axes_ok.values()))
    return {
        "porosity": eps,
        "percolates": axes_ok,
        "certified_rev": certified,
    }


def generate_xct_subvolume_splits(
    volume_shape_zyx: Tuple[int, int, int] = (2110, 2510, 2510),
    voxel_size_um: float = 0.8125,
    block_size_vox: int = 256,
    guard_band_um: float = 36.0,
    radial_valid_fraction: float = 0.82,
) -> List[BoundingBox3D]:
    """Partition cylindrical non-cupping interior into train/val/test 3D slabs separated by >= guard_band_um.

    Using axial z-slabs separated by explicit physical guard bands prevents ALL 2.5D orthogonal-slice
    (XY/XZ/YZ) and 3D PSF blur data leakage across train, validation, and test splits.
    """
    nz, ny, nx = volume_shape_zyx
    guard_vox = int(np.ceil(guard_band_um / voxel_size_um))

    # Keep blocks safely inside cylindrical non-cupping interior
    cy, cx = ny // 2, nx // 2
    r_valid = int(min(cy, cx) * radial_valid_fraction / np.sqrt(2.0))
    y_min, y_max = cy - r_valid, cy + r_valid
    x_min, x_max = cx - r_valid, cx + r_valid

    # Axial allocation: [margin | train slabs | guard | val slab | guard | test slab | margin]
    z_margin = max(64, nz // 16)
    test_z1 = nz - z_margin
    test_z0 = test_z1 - block_size_vox
    val_z1 = test_z0 - guard_vox
    val_z0 = val_z1 - block_size_vox
    train_z1 = val_z0 - guard_vox
    train_z0 = z_margin

    if train_z1 - train_z0 < block_size_vox:
        raise ValueError(
            f"Volume z={nz} too small for block_size={block_size_vox} and guard_vox={guard_vox}"
        )

    boxes: List[BoundingBox3D] = []

    def _tile_xy(split_name: str, z0: int, z1: int, prefix: str) -> None:
        idx = 0
        y = y_min
        while y + block_size_vox <= y_max:
            x = x_min
            while x + block_size_vox <= x_max:
                boxes.append(
                    BoundingBox3D(
                        split=split_name,
                        block_id=f"{prefix}_{idx:02d}",
                        z0=int(z0),
                        z1=int(z1),
                        y0=int(y),
                        y1=int(y + block_size_vox),
                        x0=int(x),
                        x1=int(x + block_size_vox),
                        voxel_size_um=voxel_size_um,
                    )
                )
                idx += 1
                x += block_size_vox + guard_vox // 2
            y += block_size_vox + guard_vox // 2

    z = train_z0
    slab_idx = 0
    while z + block_size_vox <= train_z1:
        _tile_xy("train", z, z + block_size_vox, f"train_z{slab_idx}")
        z += block_size_vox
        slab_idx += 1

    _tile_xy("val", val_z0, val_z1, "val")
    _tile_xy("test", test_z0, test_z1, "test")
    return boxes


def generate_t3_xrdct_splits(
    all_50um_scan_ids: Sequence[int] = tuple(list(range(43336, 43348)) + list(range(43393, 43401))),
    matched_50um_indices_for_25um: Sequence[int] = (9, 14, 17, 18),
    axial_step_um: float = 50.0,
    axial_guard_um: float = 100.0,
) -> Dict[str, Any]:
    """Generate T3 chemical fusion splits enforcing a +/-100 um axial z-guard band around held-out 25 um slices."""
    guard_slices = int(np.ceil(axial_guard_um / axial_step_um))
    forbidden_indices = set()
    for idx in matched_50um_indices_for_25um:
        for offset in range(-guard_slices, guard_slices + 1):
            forbidden_indices.add(idx + offset)

    allowed_self_sup_50um: List[int] = []
    excluded_guard_50um: List[int] = []
    for i, sid in enumerate(all_50um_scan_ids):
        if i in forbidden_indices:
            excluded_guard_50um.append(int(sid))
        else:
            allowed_self_sup_50um.append(int(sid))

    return {
        "held_out_test_25um_scans": [43322, 43323, 43324, 43325],
        "matched_50um_slice_indices": list(matched_50um_indices_for_25um),
        "axial_guard_band_um": axial_guard_um,
        "guard_slices_50um": guard_slices,
        "train_self_supervised_50um_scans": allowed_self_sup_50um,
        "excluded_guard_band_50um_scans": excluded_guard_50um,
    }


def generate_benchmark_splits(out_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Build and verify canonical v1 splits across T1/T2/T4 3D sub-volumes and T3 correlative slices."""
    t1_08125 = generate_xct_subvolume_splits(
        volume_shape_zyx=(2110, 2510, 2510),
        voxel_size_um=0.8125,
        block_size_vox=256,
        guard_band_um=36.0,
    )
    ok_08, min_sep_08 = verify_no_spatial_leakage(t1_08125, min_guard_band_um=32.0)
    if not ok_08:
        raise RuntimeError(f"Spatial leakage detected in 0.8125 um splits (min_sep={min_sep_08} um)")

    t1_0325 = generate_xct_subvolume_splits(
        volume_shape_zyx=(2110, 2510, 2510),
        voxel_size_um=0.325,
        block_size_vox=256,
        guard_band_um=36.0,
        radial_valid_fraction=0.78,
    )
    ok_03, min_sep_03 = verify_no_spatial_leakage(t1_0325, min_guard_band_um=32.0)
    if not ok_03:
        raise RuntimeError(f"Spatial leakage detected in 0.325 um splits (min_sep={min_sep_03} um)")

    t2_diad = generate_xct_subvolume_splits(
        volume_shape_zyx=(2160, 2560, 2560),
        voxel_size_um=0.54,
        block_size_vox=256,
        guard_band_um=36.0,
    )
    ok_diad, min_sep_diad = verify_no_spatial_leakage(t2_diad, min_guard_band_um=32.0)
    if not ok_diad:
        raise RuntimeError(f"Spatial leakage detected in DIAD 0.54 um splits (min_sep={min_sep_diad} um)")

    payload: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "min_spatial_guard_band_um": 32.0,
        "verified_min_separations_um": {
            "i13_0p8125um": round(min_sep_08, 3),
            "i13_0p325um": round(min_sep_03, 3),
            "diad_0p54um": round(min_sep_diad, 3),
        },
        "t1_0p8125um_hr_blocks": [asdict(b) for b in t1_08125],
        "t1_0p325um_hr_blocks": [asdict(b) for b in t1_0325],
        "t2_t4_diad_0p54um_blocks": [asdict(b) for b in t2_diad],
        "t3_correlative_fusion": generate_t3_xrdct_splits(),
    }

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate leakage-free 3D benchmark splits.")
    parser.add_argument("--out", type=Path, default=Path("splits/benchmark_splits_v1.json"))
    args = parser.parse_args(argv)
    splits = generate_benchmark_splits(args.out)
    print(
        json.dumps(
            {
                "saved_to": str(args.out),
                "verified_min_separations_um": splits["verified_min_separations_um"],
                "num_t1_0p8125_blocks": len(splits["t1_0p8125um_hr_blocks"]),
                "num_t3_allowed_50um_scans": len(splits["t3_correlative_fusion"]["train_self_supervised_50um_scans"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
