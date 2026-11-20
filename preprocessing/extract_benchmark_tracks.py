"""Extract Primary (`raw_float32`), Secondary (`cleaned_dering_v1` + diff maps), and `lite` sub-volumes on SCARF/Hamilton.

Implements the Zn-13X-Bench two-track derived-data extraction pipeline:
1. Primary Track (`raw_float32`): Registered, uncleaned 32-bit float sub-volumes resampled strictly with the frozen
   7-DOF transforms in `configs/i13_benchmark_transforms.json` above the 500 um ruby sphere shadow.
2. Cleaned Track (`cleaned_dering_v1`): Exact same registered float32 sub-volumes processed with frozen conservative
   polar de-ringing (`remove_rings_polar_conservative`), saved alongside the exact subtracted ring field
   (`diff_ring = raw_float32 - cleaned_dering_v1`) for 100% lossless reversibility.
   (Polynomial cupping detrending and axial z-flattening are excluded to preserve real Zn/Na chemistry.)
3. Lite Preview & OME-Zarr Layout Metadata: Writes 4x-downsampled float32 preview arrays and OME-NGFF v0.4
   coordinate metadata JSON for zero-download inspection and Hugging Face / Zenodo `IsDerivedFrom` hosting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
from basecoat import AxisSpec, coarsen_axes
from skimage.transform import downscale_local_mean

from baselines.integrations.openlsrx_adapter import resample_with_benchmark_transform
from preprocessing.artifact_cleanup import remove_rings_polar_conservative
from registration.diad_correlative import read_savu_or_zyx_slab


def build_ome_ngff_multiscales_attrs(
    name: str,
    voxel_size_um: float,
    track_version: str = "raw_float32",
) -> Dict[str, Any]:
    """Build OME-NGFF v0.4 `.zattrs` coordinate transformation metadata using basecoat's cell-centre geometry."""
    v = float(voxel_size_um)
    axes_lvl0 = tuple(AxisSpec(name=ax, kind="space", unit="um", scale=v, offset=0.0) for ax in ("z", "y", "x"))
    axes_lite4x = coarsen_axes(axes_lvl0, 4)
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": name,
                "axes": [
                    {"name": ax.name, "type": "space", "unit": "micrometer"} for ax in axes_lvl0
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [float(ax.scale) for ax in axes_lvl0]}
                        ],
                    },
                    {
                        "path": "lite_4x",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [float(ax.scale) for ax in axes_lite4x]},
                            {"type": "translation", "translation": [float(ax.offset) for ax in axes_lite4x]},
                        ],
                    },
                ],
                "metadata": {
                    "benchmark": "Zn-13X-Bench",
                    "reference_track": track_version,
                    "is_derived_from_doi": "10.1038/s41597-025-04605-9",
                },
            }
        ]
    }


def extract_paired_subvolume_tracks(
    lr_zyx: np.ndarray,
    hr_zyx: np.ndarray,
    lr_nominal_voxel_um: float,
    hr_nominal_voxel_um: float,
    ruby_calibrated_hr_voxel_um: float,
    track_name: str,
    out_h5_path: Path | str,
    hr_ring_center_yx: Optional[Tuple[float, float]] = None,
    lr_ring_center_yx: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Write paired primary (`raw_float32`), cleaned (`cleaned_dering_v1` + diff), and `lite_4x` datasets to HDF5."""
    hr_raw = np.asarray(hr_zyx, dtype=np.float32)
    lr_raw = np.asarray(lr_zyx, dtype=np.float32)

    lr_on_hr_raw, tfm_meta = resample_with_benchmark_transform(
        lr_zyx=lr_raw,
        hr_shape_zyx=hr_raw.shape,
        lr_nominal_voxel_um=lr_nominal_voxel_um,
        hr_nominal_voxel_um=hr_nominal_voxel_um,
        track=track_name,
    )

    hr_clean, rep_hr = remove_rings_polar_conservative(hr_raw, center_yx=hr_ring_center_yx)
    lr_clean_src, rep_lr = remove_rings_polar_conservative(lr_raw, center_yx=lr_ring_center_yx)
    lr_on_hr_clean, _ = resample_with_benchmark_transform(
        lr_zyx=lr_clean_src,
        hr_shape_zyx=hr_raw.shape,
        lr_nominal_voxel_um=lr_nominal_voxel_um,
        hr_nominal_voxel_um=hr_nominal_voxel_um,
        track=track_name,
    )

    diff_hr = (hr_raw - hr_clean).astype(np.float32)
    diff_lr = (lr_on_hr_raw - lr_on_hr_clean).astype(np.float32)
    hr_lite = downscale_local_mean(hr_raw, (4, 4, 4)).astype(np.float32)
    lr_lite = downscale_local_mean(lr_on_hr_raw, (4, 4, 4)).astype(np.float32)

    out_p = Path(out_h5_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ngff_attrs = build_ome_ngff_multiscales_attrs(
        name=f"zn13x_t1_{track_name}",
        voxel_size_um=ruby_calibrated_hr_voxel_um,
    )

    with h5py.File(out_p, "w") as h5f:
        g_prim = h5f.create_group("primary_raw_float32")
        g_prim.create_dataset("hr", data=hr_raw, compression="gzip", chunks=True)
        g_prim.create_dataset("lr_on_hr", data=lr_on_hr_raw, compression="gzip", chunks=True)

        g_cln = h5f.create_group("cleaned_dering_v1")
        g_cln.create_dataset("hr", data=hr_clean, compression="gzip", chunks=True)
        g_cln.create_dataset("lr_on_hr", data=lr_on_hr_clean, compression="gzip", chunks=True)
        g_cln.create_dataset("diff_hr_rings", data=diff_hr, compression="gzip", chunks=True)
        g_cln.create_dataset("diff_lr_rings", data=diff_lr, compression="gzip", chunks=True)

        g_lite = h5f.create_group("lite_4x")
        g_lite.create_dataset("hr", data=hr_lite, compression="gzip")
        g_lite.create_dataset("lr_on_hr", data=lr_lite, compression="gzip")

        h5f.attrs["ome_ngff_zattrs_json"] = json.dumps(ngff_attrs)
        h5f.attrs["transform_metadata_json"] = json.dumps(tfm_meta)

    return {
        "output_h5": str(out_p),
        "shape_zyx": list(hr_raw.shape),
        "lite_4x_shape_zyx": list(hr_lite.shape),
        "hr_dering_report": rep_hr.to_dict(),
        "lr_dering_report": rep_lr.to_dict(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract Zn-13X-Bench float32 primary & cleaned tracks on SCARF/Hamilton.")
    parser.add_argument("--lr-h5", type=Path, required=True, help="Full float32 LR HDF5 path on SCARF/Hamilton.")
    parser.add_argument("--hr-h5", type=Path, required=True, help="Full float32 HR HDF5 path on SCARF/Hamilton.")
    parser.add_argument("--track", default="2p026x", help="T1 ladder track (e.g. 2p026x or 8p101x).")
    parser.add_argument("--lr-vox-um", type=float, default=1.625)
    parser.add_argument("--hr-vox-um", type=float, default=0.8125)
    parser.add_argument("--cal-hr-vox-um", type=float, default=0.8006)
    parser.add_argument("--z0", type=int, default=640)
    parser.add_argument("--z1", type=int, default=896)
    parser.add_argument("--out-h5", type=Path, required=True)
    args = parser.parse_args(argv)

    hr_slab, _, _ = read_savu_or_zyx_slab(args.hr_h5, z0=args.z0, z1=args.z1)
    lr_z0 = max(0, int(round(args.z0 * args.hr_vox_um / args.lr_vox_um)) - 16)
    lr_z1 = int(round(args.z1 * args.hr_vox_um / args.lr_vox_um)) + 16
    lr_slab, _, _ = read_savu_or_zyx_slab(args.lr_h5, z0=lr_z0, z1=lr_z1)

    summary = extract_paired_subvolume_tracks(
        lr_zyx=lr_slab,
        hr_zyx=hr_slab,
        lr_nominal_voxel_um=args.lr_vox_um,
        hr_nominal_voxel_um=args.hr_vox_um,
        ruby_calibrated_hr_voxel_um=args.cal_hr_vox_um,
        track_name=args.track,
        out_h5_path=args.out_h5,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
