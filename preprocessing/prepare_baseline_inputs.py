"""Materialize split-sliced TIFF & calibrated `.ome.zarr` baseline inputs from Stage 0 tracks (`ECOSYSTEM.md` Layer 4).

Bridges `scripts/slurm/01_stage0_audit_and_extract.sh` directly to:
  - `scripts/slurm/02_t1_headline_array.sh` (`data/benchmarks/zn13x_float32/`):
      * `<track>_<regime>_train_lr.tif` & `<track>_<regime>_train_hr.tif` across
        `{synthetic_clean, synthetic_noisy, synthetic_jittered, real_paired}`
      * `<track>_real_paired_test_lr.tif` & `<track>_real_paired_test_hr.ome.zarr` (calibrated `basecoat.Volume`)
  - `scripts/slurm/03_t2_loso_array.sh` (`data/benchmarks/zn13x_t2_diad/`):
      * `zn_50um_<scan>.tif`, `na_50um_<scan>.tif`, `stack_zn_na_25um_<scan>.tif`
      * `xct_slab_z<zz>_<scan>.tif` (0.54 um DIAD XCT slab averaged over the 25 um pencil-beam height)
      * `ref_25um_<scan>.ome.zarr` (calibrated 2-channel `(Zn, Na)` OME-Zarr v0.4 reference store)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tifffile
from basecoat import AxisSpec, Volume, open_savu_h5, write_ome_zarr, write_volume
from scipy.ndimage import zoom

from baselines.integrations.openlsrx_adapter import (
    load_benchmark_basecoat_transform,
    synthesize_degraded_lr_from_hr,
)
from baselines.integrations.xfuse_adapter import (
    DEFAULT_SPLITS_PATH,
    load_xfuse_benchmark_t2_bundle,
)


def prepare_t1_baseline_inputs(
    extracted_h5: Path | str,
    out_dir: Path | str,
    *,
    track: str = "2p026x",
    ruby_calibrated_hr_voxel_um: float = 0.8006,
    reference_track: str = "primary_raw_float32",
    splits_path: Path | str = DEFAULT_SPLITS_PATH,
    seed: int = 42,
) -> dict[str, Any]:
    """Slice an extracted paired track HDF5 into non-overlapping train/test slabs and write T1 TIFF/.ome.zarr files."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    _, meta = load_benchmark_basecoat_transform(track=track)
    true_ratio = float(meta["true_magnification_ratio"])
    lr_vox_um = float(ruby_calibrated_hr_voxel_um) * true_ratio

    with h5py.File(extracted_h5, "r") as fh:
        grp = fh[reference_track]
        hr_full = np.asarray(grp["hr"][:], dtype=np.float32)
        lr_on_hr_full = np.asarray(grp["lr_on_hr"][:], dtype=np.float32)

    nz = int(hr_full.shape[0])
    # Respect LR-voxel guard band between axial train and test slabs even on compact slabs
    splits_doc = json.loads(Path(splits_path).read_text(encoding="utf-8")) if Path(splits_path).is_file() else {}
    buf_key = "t1_8p101x_169068_to_169065" if "8p101" in track else "t1_2p026x_169067_to_169066"
    buf_lr_vox = int(splits_doc.get("lr_voxel_buffers", {}).get(buf_key, {}).get("buffer_lr_vox", 48))
    guard_hr_vox = min(max(2, nz // 6), int(round(buf_lr_vox * true_ratio)))
    train_end = max(4, (nz - guard_hr_vox) * 2 // 3)
    test_start = min(nz - 4, train_end + guard_hr_vox)
    if test_start <= train_end:
        train_end = max(2, nz // 2 - 1)
        test_start = min(nz - 2, train_end + 2)

    hr_train = np.ascontiguousarray(hr_full[:train_end], dtype=np.float32)
    lr_on_hr_train = np.ascontiguousarray(lr_on_hr_full[:train_end], dtype=np.float32)
    hr_test = np.ascontiguousarray(hr_full[test_start:], dtype=np.float32)
    lr_on_hr_test = np.ascontiguousarray(lr_on_hr_full[test_start:], dtype=np.float32)

    inv_scale = 1.0 / true_ratio
    real_train_lr = zoom(lr_on_hr_train, zoom=(inv_scale, inv_scale, inv_scale), order=1, prefilter=False).astype(
        np.float32
    )
    real_test_lr = zoom(lr_on_hr_test, zoom=(inv_scale, inv_scale, inv_scale), order=1, prefilter=False).astype(
        np.float32
    )

    regimes: dict[str, np.ndarray] = {
        "synthetic_clean": synthesize_degraded_lr_from_hr(
            hr_train, true_scale_ratio=true_ratio, with_noise=False, registration_jitter_hr_vox=0.0, seed=seed
        ),
        "synthetic_noisy": synthesize_degraded_lr_from_hr(
            hr_train, true_scale_ratio=true_ratio, with_noise=True, registration_jitter_hr_vox=0.0, seed=seed
        ),
        "synthetic_jittered": synthesize_degraded_lr_from_hr(
            hr_train, true_scale_ratio=true_ratio, with_noise=True, registration_jitter_hr_vox=2.5, seed=seed
        ),
        "real_paired": real_train_lr,
    }

    written_files: dict[str, str] = {}
    for rname, lr_arr in regimes.items():
        lr_path = dest / f"{track}_{rname}_train_lr.tif"
        hr_path = dest / f"{track}_{rname}_train_hr.tif"
        write_volume(Volume.from_array(lr_arr, voxel_size_um=lr_vox_um), lr_path)
        write_volume(Volume.from_array(hr_train, voxel_size_um=ruby_calibrated_hr_voxel_um), hr_path)
        written_files[f"{rname}_train_lr"] = str(lr_path)
        written_files[f"{rname}_train_hr"] = str(hr_path)

    test_lr_path = dest / f"{track}_real_paired_test_lr.tif"
    test_hr_zarr = dest / f"{track}_real_paired_test_hr.ome.zarr"
    write_volume(Volume.from_array(real_test_lr, voxel_size_um=lr_vox_um), test_lr_path)
    ref_tag = "raw_float32" if "raw" in reference_track else "cleaned_dering_v1"
    write_ome_zarr(
        Volume.from_array(
            hr_test,
            voxel_size_um=ruby_calibrated_hr_voxel_um,
            space=f"i13_{meta.get('hr_scan', 'hr')}",
            provenance={"zn13x_bench": {"reference_track": ref_tag, "track": track}},
        ),
        test_hr_zarr,
        multiscale_levels=1,
        overwrite=True,
    )
    written_files["real_paired_test_lr"] = str(test_lr_path)
    written_files["real_paired_test_hr_ome_zarr"] = str(test_hr_zarr)

    return {
        "track": track,
        "train_hr_shape": list(hr_train.shape),
        "test_hr_shape": list(hr_test.shape),
        "guard_hr_vox": int(guard_hr_vox),
        "files": written_files,
    }


def prepare_t2_baseline_inputs(
    xrdct_dir: Path | str,
    out_dir: Path | str,
    *,
    diad_xct_h5: Path | str | None = None,
    splits_path: Path | str = DEFAULT_SPLITS_PATH,
) -> dict[str, Any]:
    """Materialize Task T2 4-fold LOSO 50 um guides, 25 um `.ome.zarr` references, and optional DIAD XCT slabs."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    splits_raw = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    folds = list(splits_raw["t2_chemical_fusion_loso"]["loso_folds"].values())

    bundle_all = load_xfuse_benchmark_t2_bundle(xrdct_dir=xrdct_dir, splits_path=splits_path, loso_fold=None)
    all_50_ids: list[int] = [int(s) for s in bundle_all["scan_ids_50um"]]
    zn_50_zyx = np.asarray(bundle_all["guide_50um_zyx"]["Zn"], dtype=np.float32)
    na_50_zyx = np.asarray(bundle_all["guide_50um_zyx"]["Na"], dtype=np.float32)

    for idx, sid50 in enumerate(all_50_ids):
        zn_p = dest / f"zn_50um_{sid50}.tif"
        na_p = dest / f"na_50um_{sid50}.tif"
        tifffile.imwrite(zn_p, zn_50_zyx[idx], dtype=np.float32)
        tifffile.imwrite(na_p, na_50_zyx[idx], dtype=np.float32)
        written.extend([str(zn_p), str(na_p)])

    for fold in folds:
        fidx = int(fold["fold_index"])
        fb = load_xfuse_benchmark_t2_bundle(xrdct_dir=xrdct_dir, splits_path=splits_path, loso_fold=fidx)
        for sid25, maps in {**fb["train_25um_slices"], **fb["held_out_test_25um"]}.items():
            stack_cyx = np.stack(
                [np.asarray(maps["zn"], dtype=np.float32), np.asarray(maps["na"], dtype=np.float32)],
                axis=0,
            )
            st_tif = dest / f"stack_zn_na_25um_{int(sid25)}.tif"
            if not st_tif.exists():
                tifffile.imwrite(st_tif, stack_cyx, dtype=np.float32)
                written.append(str(st_tif))

            ref_zarr = dest / f"ref_25um_{int(sid25)}.ome.zarr"
            if not ref_zarr.exists():
                axes = (
                    AxisSpec(name="c", kind="channel", unit="none"),
                    AxisSpec(name="y", kind="space", unit="um", scale=25.0),
                    AxisSpec(name="x", kind="space", unit="um", scale=25.0),
                )
                write_ome_zarr(
                    Volume(
                        stack_cyx,
                        axes=axes,
                        space="diad_xrdct_25um",
                        provenance={"zn13x_bench": {"reference_track": "raw_float32", "scan_25um": int(sid25)}},
                    ),
                    ref_zarr,
                    multiscale_levels=1,
                    overwrite=True,
                )
                written.append(str(ref_zarr))

        if diad_xct_h5 is not None and Path(diad_xct_h5).is_file():
            z_idx = int(fold["test_matched_50um_z_index"])
            sid_hold = int(fold["test_25um_scan"])
            with open_savu_h5(diad_xct_h5, voxel_size_um=0.54, space="diad_xct_0p54um") as lv:
                nz_xct = int(lv.shape[0])
                # Map 50 um slice index (0..19) across axial span with 25 um (~46 slice) pencil-beam boxcar average
                z_c = int(round((z_idx + 0.5) / 20.0 * nz_xct))
                z_lo = max(0, z_c - 23)
                z_hi = min(nz_xct, z_c + 23)
                slab_vol = lv.read_region((slice(z_lo, max(z_lo + 1, z_hi)), slice(None), slice(None)))
                slab_2d = np.mean(np.asarray(slab_vol.data, dtype=np.float32), axis=0)
            xct_tif = dest / f"xct_slab_z{z_idx:02d}_{sid_hold}.tif"
            tifffile.imwrite(xct_tif, slab_2d, dtype=np.float32)
            written.append(str(xct_tif))

    if diad_xct_h5 is not None and Path(diad_xct_h5).is_file():
        sub_h5 = dest / "diad_xct_subvolume.h5"
        if not sub_h5.exists():
            with open_savu_h5(diad_xct_h5, voxel_size_um=0.54, space="diad_xct_0p54um") as lv:
                sub = lv.read_region((slice(None, None, 4), slice(None, None, 2), slice(None, None, 2)))
            with h5py.File(sub_h5, "w") as h5f:
                h5f.create_dataset("data", data=np.asarray(sub.data, dtype=np.float32), compression="gzip")
            written.append(str(sub_h5))

    return {"out_dir": str(dest), "num_written": len(written), "written": written}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare split-sliced TIFF and calibrated .ome.zarr inputs for Task T1 (OpenLSR-X) and Task T2 (X-FUSE)."
    )
    parser.add_argument("--mode", choices=["t1", "t2"], required=True)
    parser.add_argument("--extracted-h5", type=Path, default=None, help="Paired track HDF5 from extract_benchmark_tracks (mode=t1).")
    parser.add_argument("--track", default="2p026x", help="T1 magnification track (default: 2p026x).")
    parser.add_argument("--cal-hr-vox-um", type=float, default=0.8006, help="Ruby-calibrated HR voxel size in um.")
    parser.add_argument("--xrdct-dir", type=Path, default=Path("data/cache/zenodo"), help="Directory with output_*_{na,zn}.nxs (mode=t2).")
    parser.add_argument("--diad-xct-h5", type=Path, default=None, help="Optional DIAD 0.54 um XCT HDF5 path (mode=t2).")
    parser.add_argument("--splits-path", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.mode == "t1":
        if args.extracted_h5 is None:
            parser.error("--extracted-h5 is required when --mode=t1.")
        res = prepare_t1_baseline_inputs(
            extracted_h5=args.extracted_h5,
            out_dir=args.out_dir,
            track=args.track,
            ruby_calibrated_hr_voxel_um=args.cal_hr_vox_um,
            splits_path=args.splits_path,
            seed=args.seed,
        )
    else:
        res = prepare_t2_baseline_inputs(
            xrdct_dir=args.xrdct_dir,
            out_dir=args.out_dir,
            diad_xct_h5=args.diad_xct_h5,
            splits_path=args.splits_path,
        )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
