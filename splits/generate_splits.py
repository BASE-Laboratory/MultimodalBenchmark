"""Leakage-free 3D sub-volume split generator with LR-voxel buffers & 4-fold LOSO XRD-CT CV (Zn-13X-Bench v2.0).

Implements the revised Zn-13X-Bench split rules:
1. T1 XCT Super-Resolution: Disjoint axial slabs strictly ABOVE the 500 um ruby sphere's pink-beam shadow,
   separated by an axial buffer of at least `patch_size_lr_vox + receptive_field_lr_vox` measured in
   LOW-RESOLUTION voxels (solving the flaw where a fixed 20 um buffer is only 8 voxels at 2.6 um!).
   Explicitly documents the axial Zn->Na chemistry shift across train/val/test slabs as an out-of-distribution
   generalization test.
2. T2 Chemical Fusion: 4-fold Leave-One-Slice-Out (LOSO) cross-validation across all four 25 um heights
   (`43324@z9`, `43322@z14`, `43323@z17`, `43325@z18`), each enforcing a +/-100 um axial z-guard band on 50 um slices.
"""

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
    ok_rev = bool(eps >= min_porosity and all(axes_ok.values()))
    return {
        "porosity": eps,
        "percolates": axes_ok,
        "certified_rev": ok_rev,
    }


def compute_lr_voxel_buffer_um(
    lr_voxel_um: float,
    patch_z_lr_vox: int = 24,
    receptive_field_lr_vox: int = 16,
    min_physical_um: float = 36.0,
) -> Tuple[int, float]:
    """Compute required axial split buffer in LR voxels (`patch_z_lr + receptive_field_lr`) and physical um."""
    buf_lr_vox = int(patch_z_lr_vox) + int(receptive_field_lr_vox)
    buf_um = max(float(buf_lr_vox) * float(lr_voxel_um), float(min_physical_um))
    buf_lr_vox = int(np.ceil(buf_um / float(lr_voxel_um)))
    return buf_lr_vox, round(float(buf_lr_vox) * float(lr_voxel_um), 3)


def generate_xct_subvolume_splits(
    volume_shape_zyx: Tuple[int, int, int] = (2110, 2510, 2510),
    voxel_size_um: float = 0.8125,
    block_size_vox: int = 256,
    guard_band_um: float = 36.0,
    radial_valid_fraction: float = 0.82,
    ruby_shadow_z_min_vox: int = 0,
    lr_voxel_um: Optional[float] = None,
    patch_z_lr_vox: int = 24,
    receptive_field_lr_vox: int = 16,
) -> List[BoundingBox3D]:
    """Partition interior above the ruby shadow into train/val/test slabs with LR-voxel-governed guard bands."""
    nz, ny, nx = volume_shape_zyx
    eff_guard_um = float(guard_band_um)
    if lr_voxel_um is not None and lr_voxel_um > 0:
        _, lr_buf_um = compute_lr_voxel_buffer_um(
            lr_voxel_um=lr_voxel_um,
            patch_z_lr_vox=patch_z_lr_vox,
            receptive_field_lr_vox=receptive_field_lr_vox,
            min_physical_um=guard_band_um,
        )
        eff_guard_um = max(eff_guard_um, lr_buf_um)

    guard_vox = int(np.ceil(eff_guard_um / voxel_size_um))

    cy, cx = ny // 2, nx // 2
    r_valid = int(min(cy, cx) * radial_valid_fraction / np.sqrt(2.0))
    y_min, y_max = cy - r_valid, cy + r_valid
    x_min, x_max = cx - r_valid, cx + r_valid

    z_margin_top = max(64, nz // 16)
    z_margin_bot = max(int(ruby_shadow_z_min_vox), max(64, nz // 16))
    test_z1 = nz - z_margin_top
    test_z0 = test_z1 - block_size_vox
    val_z1 = test_z0 - guard_vox
    val_z0 = val_z1 - block_size_vox
    train_z1 = val_z0 - guard_vox
    train_z0 = z_margin_bot

    if train_z1 - train_z0 < block_size_vox:
        raise ValueError(
            f"Volume z={nz} (z_bot={train_z0}) too small for block_size={block_size_vox} and guard_vox={guard_vox}"
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


def generate_t2_loso_xrdct_splits(
    all_50um_scan_ids: Sequence[int] = tuple(list(range(43336, 43348)) + list(range(43393, 43401))),
    scan_to_z_index: Optional[Dict[int, int]] = None,
    axial_step_um: float = 50.0,
    axial_guard_um: float = 100.0,
) -> Dict[str, Any]:
    """Generate 4-fold Leave-One-Slice-Out (LOSO) CV splits for Task T2 chemical super-resolution & fusion."""
    mapping = scan_to_z_index or {43324: 9, 43322: 14, 43323: 17, 43325: 18}
    guard_slices = int(np.ceil(axial_guard_um / axial_step_um))
    ordered_25 = [43324, 43322, 43323, 43325]

    loso_folds: Dict[str, Any] = {}
    for fold_idx, test_sid in enumerate(ordered_25):
        test_z = int(mapping[test_sid])
        train_25 = [s for s in ordered_25 if s != test_sid]
        forbidden_50_idx = set(range(test_z - guard_slices, test_z + guard_slices + 1))
        allowed_50 = [int(sid) for i, sid in enumerate(all_50um_scan_ids) if i not in forbidden_50_idx]
        excluded_50 = [int(sid) for i, sid in enumerate(all_50um_scan_ids) if i in forbidden_50_idx]
        loso_folds[f"fold_{fold_idx}_holdout_{test_sid}_z{test_z}"] = {
            "fold_index": fold_idx,
            "test_25um_scan": int(test_sid),
            "test_matched_50um_z_index": test_z,
            "test_matched_50um_scan": int(all_50um_scan_ids[test_z]),
            "train_25um_scans": train_25,
            "allowed_self_supervised_50um_scans": allowed_50,
            "excluded_guard_band_50um_scans": excluded_50,
        }

    # Also compute global self-supervised 50 um set excluding +/-100 um around ALL four 25 um slices
    all_forbidden = set()
    for z_i in mapping.values():
        all_forbidden.update(range(int(z_i) - guard_slices, int(z_i) + guard_slices + 1))
    global_allowed_50 = [int(sid) for i, sid in enumerate(all_50um_scan_ids) if i not in all_forbidden]
    global_excluded_50 = [int(sid) for i, sid in enumerate(all_50um_scan_ids) if i in all_forbidden]

    return {
        "evaluation_protocol": "leave_one_slice_out_4fold_cv",
        "held_out_test_25um_scans": [43322, 43323, 43324, 43325],
        "matched_50um_slice_indices": [mapping[s] for s in [43324, 43322, 43323, 43325]],
        "axial_guard_band_um": axial_guard_um,
        "guard_slices_50um": guard_slices,
        "loso_folds": loso_folds,
        "train_self_supervised_50um_scans": global_allowed_50,
        "excluded_guard_band_50um_scans": global_excluded_50,
    }


generate_t3_xrdct_splits = generate_t2_loso_xrdct_splits


def generate_benchmark_splits(out_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Build and verify canonical Zn-13X-Bench splits with LR-voxel buffers and 4-fold T2 LOSO CV."""
    # Track 2.026x (LR = 169067 at 1.6212 um -> HR = 169066 at 0.8006 um):
    # Patch = 32 LR vox + Receptive Field = 16 LR vox -> 48 LR vox buffer = 77.8 um (98 HR voxels)!
    # Ruby sphere in full 2110-slice 169066 sits below z=620; start train slab strictly above ruby shadow!
    buf_vox_2x, buf_um_2x = compute_lr_voxel_buffer_um(1.6212, patch_z_lr_vox=32, receptive_field_lr_vox=16)
    t1_2p026x = generate_xct_subvolume_splits(
        volume_shape_zyx=(2110, 2510, 2510),
        voxel_size_um=0.8006,
        block_size_vox=256,
        guard_band_um=buf_um_2x,
        ruby_shadow_z_min_vox=640,
        lr_voxel_um=1.6212,
        patch_z_lr_vox=32,
        receptive_field_lr_vox=16,
    )
    ok_08, min_sep_08 = verify_no_spatial_leakage(t1_2p026x, min_guard_band_um=buf_um_2x - 1.0)
    if not ok_08:
        raise RuntimeError(f"Spatial leakage in 2.026x splits (min_sep={min_sep_08} um < {buf_um_2x} um)")

    # Track 8.101x (LR = 169068 at 2.5945 um -> HR = 169065 at 0.3211 um):
    # Entire 2110-slice HR volume spans ~260 LR z-slices (677 um, wholly above the ruby sphere!).
    # Use anisotropic LR z-patch = 24 LR vox + RF = 16 LR vox -> 40 LR vox buffer = 103.8 um (324 HR voxels)!
    buf_vox_8x, buf_um_8x = compute_lr_voxel_buffer_um(2.5945, patch_z_lr_vox=24, receptive_field_lr_vox=16)
    t1_8p101x = generate_xct_subvolume_splits(
        volume_shape_zyx=(2110, 2510, 2510),
        voxel_size_um=0.3211,
        block_size_vox=256,
        guard_band_um=buf_um_8x,
        radial_valid_fraction=0.78,
        ruby_shadow_z_min_vox=128,
        lr_voxel_um=2.5945,
        patch_z_lr_vox=24,
        receptive_field_lr_vox=16,
    )
    ok_03, min_sep_03 = verify_no_spatial_leakage(t1_8p101x, min_guard_band_um=buf_um_8x - 1.0)
    if not ok_03:
        raise RuntimeError(f"Spatial leakage in 8.101x splits (min_sep={min_sep_03} um < {buf_um_8x} um)")

    diad_blocks = generate_xct_subvolume_splits(
        volume_shape_zyx=(2160, 2560, 2560),
        voxel_size_um=0.54,
        block_size_vox=256,
        guard_band_um=78.0,
        ruby_shadow_z_min_vox=650,
    )
    ok_diad, min_sep_diad = verify_no_spatial_leakage(diad_blocks, min_guard_band_um=75.0)
    if not ok_diad:
        raise RuntimeError(f"Spatial leakage in DIAD 0.54 um splits (min_sep={min_sep_diad} um)")

    t2_loso = generate_t2_loso_xrdct_splits()
    payload: Dict[str, Any] = {
        "schema_version": "2.0.0",
        "split_buffer_policy": (
            "Axial slab buffers are defined in LOW-RESOLUTION voxels (`patch_z_lr_vox + receptive_field_lr_vox`) "
            "strictly above the 500 um ruby sphere shadow. Because Zn/Na ion exchange varies axially along the "
            "pellet (99% Zn base -> 10% Zn top), axial train/val/test slabs deliberately induce an out-of-distribution "
            "attenuation/chemistry generalization test."
        ),
        "lr_voxel_buffers": {
            "t1_2p026x_169067_to_169066": {
                "patch_z_lr_vox": 32,
                "receptive_field_lr_vox": 16,
                "buffer_lr_vox": buf_vox_2x,
                "buffer_um": buf_um_2x,
            },
            "t1_8p101x_169068_to_169065": {
                "patch_z_lr_vox": 24,
                "receptive_field_lr_vox": 16,
                "buffer_lr_vox": buf_vox_8x,
                "buffer_um": buf_um_8x,
            },
        },
        "min_spatial_guard_band_um": round(min(min_sep_08, min_sep_03, min_sep_diad), 3),
        "verified_min_separations_um": {
            "i13_2p026x_hr_0p8006um": round(min_sep_08, 3),
            "i13_8p101x_hr_0p3211um": round(min_sep_03, 3),
            "diad_0p54um": round(min_sep_diad, 3),
        },
        "t1_0p8125um_hr_blocks": [asdict(b) for b in t1_2p026x],
        "t1_0p325um_hr_blocks": [asdict(b) for b in t1_8p101x],
        "t2_t4_diad_0p54um_blocks": [asdict(b) for b in diad_blocks],
        "t2_chemical_fusion_loso": t2_loso,
        "t3_correlative_fusion": t2_loso,  # backward-compatible alias
    }

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate leakage-free Zn-13X-Bench splits.")
    parser.add_argument("--out", type=Path, default=Path("splits/benchmark_splits_v1.json"))
    args = parser.parse_args(argv)
    splits = generate_benchmark_splits(args.out)
    print(
        json.dumps(
            {
                "saved_to": str(args.out),
                "lr_voxel_buffers": splits["lr_voxel_buffers"],
                "verified_min_separations_um": splits["verified_min_separations_um"],
                "num_t1_2p026x_blocks": len(splits["t1_0p8125um_hr_blocks"]),
                "num_t1_8p101x_blocks": len(splits["t1_0p325um_hr_blocks"]),
                "num_t2_loso_folds": len(splits["t2_chemical_fusion_loso"]["loso_folds"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
