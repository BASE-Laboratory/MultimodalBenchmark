"""Zn-13X-Bench Derived-Data Release Packager (OME-Zarr v0.4 + Zenodo IsDerivedFrom + HF Lite + Croissant Check).

Implements Section 7 ('Access, format and maintenance') of the Zn-13X-Bench paper plan:
  1. Exports registered 2D/3D volumes as OME-NGFF / OME-Zarr v0.4 stores with physical `micrometer`
     `coordinateTransformations` alongside `.npz` bundles for `evaluate.py`.
  2. Writes `zenodo_derived_metadata.json` with `isDerivedFrom` links to all upstream Green et al. (2025)
     Zenodo DOIs so upstream raw data is never duplicated.
  3. Builds a lightweight `< 2 GB` `lite/` subset (`zn13x_bench_lite`) for Hugging Face Hub mirroring & CI smoke tests.
  4. Validates `configs/croissant.json` against MLCommons Croissant 1.0 JSON-LD structural requirements.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr
from numcodecs import Blosc
from skimage.transform import downscale_local_mean

UPSTREAM_ZENODO_DOIS: Tuple[Tuple[str, str], ...] = (
    ("10.1038/s41597-025-04605-9", "isSupplementTo"),
    ("10.5281/zenodo.12206815", "isDerivedFrom"),
    ("10.5281/zenodo.13327651", "isDerivedFrom"),
    ("10.5281/zenodo.13327682", "isDerivedFrom"),
    ("10.5281/zenodo.13327692", "isDerivedFrom"),
    ("10.5281/zenodo.13349402", "isDerivedFrom"),
    ("10.5281/zenodo.13329639", "isDerivedFrom"),
    ("10.5281/zenodo.13329670", "isDerivedFrom"),
)


def write_ome_zarr_v04(
    array: np.ndarray,
    store_path: Path | str,
    voxel_size_um: Sequence[float],
    axes_names: Optional[Sequence[str]] = None,
    channel_names: Optional[Sequence[str]] = None,
    multiscale_levels: int = 2,
    extra_attrs: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a 2D `(y, x)`, 3D `(z, y, x)` or `(c, y, x)`, or 4D `(c, z, y, x)` array as OME-Zarr v0.4."""
    arr = np.asarray(array)
    out = Path(store_path)
    if out.exists():
        shutil.rmtree(out)

    ndim = arr.ndim
    if axes_names is None:
        if ndim == 2:
            axes_names = ("y", "x")
        elif ndim == 3:
            axes_names = ("z", "y", "x")
        elif ndim == 4:
            axes_names = ("c", "z", "y", "x")
        else:
            raise ValueError(f"Unsupported array ndim={ndim} for OME-Zarr v0.4")

    if len(voxel_size_um) != ndim:
        raise ValueError(f"voxel_size_um length {len(voxel_size_um)} must match array ndim {ndim}")

    axes_meta = []
    for ax in axes_names:
        if ax == "c":
            axes_meta.append({"name": "c", "type": "channel"})
        else:
            axes_meta.append({"name": ax, "type": "space", "unit": "micrometer"})

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(out), mode="w", zarr_version=2)

    datasets_meta: List[Dict[str, Any]] = []
    cur = arr
    cur_scale = [float(s) for s in voxel_size_um]

    for lvl in range(max(1, int(multiscale_levels))):
        chunks = tuple(min(s, 1 if ax == "c" else 128) for s, ax in zip(cur.shape, axes_names))
        root.create_dataset(
            str(lvl),
            data=cur,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        datasets_meta.append(
            {
                "path": str(lvl),
                "coordinateTransformations": [{"type": "scale", "scale": list(cur_scale)}],
            }
        )
        # Stop pyramid if spatial dimensions are already compact (< 32 px)
        spatial_dims = [s for s, ax in zip(cur.shape, axes_names) if ax != "c"]
        if min(spatial_dims) < 32 or lvl + 1 >= multiscale_levels:
            break
        factors = tuple(1 if ax == "c" else 2 for ax in axes_names)
        cur = downscale_local_mean(cur, factors).astype(arr.dtype)
        cur_scale = [s * f for s, f in zip(cur_scale, factors)]

    multiscales_entry: Dict[str, Any] = {
        "version": "0.4",
        "name": out.stem,
        "axes": axes_meta,
        "datasets": datasets_meta,
        "type": "local_mean_downscale",
    }
    root.attrs["multiscales"] = [multiscales_entry]
    if channel_names is not None:
        root.attrs["omero"] = {
            "name": out.stem,
            "version": "0.4",
            "channels": [{"label": str(lbl), "active": True} for lbl in channel_names],
        }
    if extra_attrs:
        root.attrs["zn13x_bench"] = dict(extra_attrs)
    return out


def validate_croissant_jsonld(croissant_path: Path | str) -> Dict[str, Any]:
    """Validate MLCommons Croissant 1.0 JSON-LD required top-level, distribution, and recordSet fields."""
    p = Path(croissant_path)
    doc = json.loads(p.read_text(encoding="utf-8"))

    errors: List[str] = []
    if doc.get("@type") != "sc:Dataset":
        errors.append("Root '@type' must be 'sc:Dataset'.")
    conforms = str(doc.get("conformsTo", ""))
    if "mlcommons.org/croissant/1.0" not in conforms:
        errors.append("Missing 'conformsTo': 'http://mlcommons.org/croissant/1.0'.")
    for req in ("name", "description", "license", "url", "creator", "citation", "distribution", "recordSet"):
        if req not in doc or not doc[req]:
            errors.append(f"Missing or empty required root property '{req}'.")

    dist_ids = set()
    for item in doc.get("distribution", []):
        oid = item.get("@id")
        otype = item.get("@type")
        if not oid or otype not in ("cr:FileObject", "cr:FileSet"):
            errors.append(f"Invalid distribution entry: {item}")
        else:
            dist_ids.add(oid)
        if "encodingFormat" not in item:
            errors.append(f"Distribution '{oid}' missing 'encodingFormat'.")

    record_sets = doc.get("recordSet", [])
    for rs in record_sets:
        if rs.get("@type") != "cr:RecordSet" or not rs.get("field"):
            errors.append(f"Invalid RecordSet '{rs.get('@id')}'.")
        for fld in rs.get("field", []):
            if fld.get("@type") != "cr:Field" or "dataType" not in fld:
                errors.append(f"Field '{fld.get('@id')}' missing '@type': 'cr:Field' or 'dataType'.")
            fo_ref = fld.get("source", {}).get("fileObject", {}).get("@id")
            fs_ref = fld.get("source", {}).get("fileSet", {}).get("@id")
            ref = fo_ref or fs_ref
            if ref and ref not in dist_ids:
                errors.append(f"Field '{fld.get('@id')}' references unknown distribution '{ref}'.")

    return {
        "valid": len(errors) == 0,
        "croissant_path": str(p),
        "num_distributions": len(dist_ids),
        "num_record_sets": len(record_sets),
        "errors": errors,
    }


def build_zenodo_derived_metadata(version: str = "2.0.0") -> Dict[str, Any]:
    """Generate Zenodo REST API `metadata` block with explicit `isDerivedFrom` provenance links."""
    return {
        "upload_type": "dataset",
        "title": "Zn-13X-Bench: Verified Multi-Resolution Synchrotron XCT & XRD-CT Pairs and Physics-Scoring Suite",
        "version": version,
        "license": "cc-by-4.0",
        "creators": [
            {"name": "Le Houx, James", "affiliation": "BASE Laboratory"},
            {"name": "Green, Calum", "affiliation": "Diamond Light Source / University College London"},
        ],
        "description": (
            "Derived benchmark release accompanying Zn-13X-Bench, built upon Green et al., Scientific Data 12, 329 "
            "(2025). Provides frozen 7-DOF similarity transforms at ruby-calibrated magnification ratios, two-track "
            "32-bit float sub-volumes (raw_float32 primary and cleaned_dering_v1 secondary with difference maps), "
            "continuum-subtracted and rotation-aligned 25 um / 50 um DIAD XRD-CT Na/Zn abundance and variance maps, "
            "LR-voxel leakage-free splits, and Croissant 1.0 metadata."
        ),
        "related_identifiers": [
            {"identifier": doi, "relation": rel, "resource_type": "dataset" if "zenodo" in doi else "publication-article"}
            for doi, rel in UPSTREAM_ZENODO_DOIS
        ],
        "keywords": [
            "synchrotron tomography",
            "super-resolution",
            "XRD-CT",
            "multimodal fusion",
            "porous media transport",
            "NeurIPS Datasets and Benchmarks",
            "Croissant",
        ],
    }


def build_derived_release_package(
    out_dir: Path | str,
    repo_root: Optional[Path | str] = None,
    xrdct_bundles_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Assemble the derived Zenodo + Hugging Face `zn13x_bench_lite` release directory."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Validate and copy Croissant 1.0, Datasheet, Frozen 7-DOF Transforms, and Splits
    croissant_src = root / "configs" / "croissant.json"
    croissant_check = validate_croissant_jsonld(croissant_src)
    if not croissant_check["valid"]:
        raise ValueError(f"Croissant 1.0 validation failed: {croissant_check['errors']}")

    (dest / "metadata").mkdir(exist_ok=True)
    (dest / "transforms").mkdir(exist_ok=True)
    (dest / "splits").mkdir(exist_ok=True)
    (dest / "lite").mkdir(exist_ok=True)

    shutil.copy2(croissant_src, dest / "metadata" / "croissant.json")
    shutil.copy2(root / "DATASHEET.md", dest / "metadata" / "DATASHEET.md")
    shutil.copy2(root / "configs" / "i13_benchmark_transforms.json", dest / "transforms" / "i13_benchmark_transforms.json")
    shutil.copy2(root / "splits" / "benchmark_splits_v1.json", dest / "splits" / "benchmark_splits_v1.json")

    zenodo_meta = build_zenodo_derived_metadata()
    (dest / "metadata" / "zenodo_derived_metadata.json").write_text(
        json.dumps(zenodo_meta, indent=2), encoding="utf-8"
    )

    exported_zarr_stores: List[str] = []

    # 2. Export calibrated 25 um & 50 um XRD-CT reference bundles as OME-Zarr v0.4 + lite package if present
    if xrdct_bundles_dir is not None and Path(xrdct_bundles_dir).exists():
        xdir = Path(xrdct_bundles_dir)
        zarr_out_dir = dest / "xrdct_calibrated_ome_zarr"
        lite_xrd_dir = dest / "lite" / "t2_xrdct_pairs"
        zarr_out_dir.mkdir(parents=True, exist_ok=True)
        lite_xrd_dir.mkdir(parents=True, exist_ok=True)

        for npz_path in sorted(xdir.glob("pair_25um_*.npz")):
            shutil.copy2(npz_path, lite_xrd_dir / npz_path.name)
            with np.load(npz_path) as d:
                # 4-channel 25 um stack: [zn_25um, na_25um, zn_sigma2, na_sigma2]
                stack_25 = np.stack(
                    [d["zn_25um"], d["na_25um"], d["zn_sigma2"], d["na_sigma2"]], axis=0
                ).astype(np.float32)
                zpath = write_ome_zarr_v04(
                    stack_25,
                    zarr_out_dir / f"{npz_path.stem}.ome.zarr",
                    voxel_size_um=(1.0, 25.0, 25.0),
                    axes_names=("c", "y", "x"),
                    channel_names=("Zn-13X_25um", "Na-13X_25um", "Zn_sigma2_odd_even", "Na_sigma2_odd_even"),
                    multiscale_levels=1,
                    extra_attrs={
                        "scan_25um": int(d["scan_25um"]),
                        "matched_50um_scan": int(d["scan_50um"]),
                        "matched_50um_z_index": int(d["z_index_50um"]),
                        "frc_sb0p5_cutoff_cyc_per_px": 0.43,
                        "reference_track": "raw_float32",
                    },
                )
                exported_zarr_stores.append(str(zpath))

    total_bytes = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    manifest_summary = {
        "release_dir": str(dest),
        "croissant_validation": croissant_check,
        "num_ome_zarr_stores": len(exported_zarr_stores),
        "ome_zarr_stores": exported_zarr_stores,
        "total_size_mb": round(total_bytes / (1024.0 * 1024.0), 3),
        "lite_under_2gb": bool(total_bytes < 2 * (1024**3)),
    }
    (dest / "release_manifest.json").write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    return manifest_summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Zn-13X-Bench derived OME-Zarr + Lite release & validate Croissant.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/derived_release_v2"))
    parser.add_argument(
        "--xrdct-bundles-dir",
        type=Path,
        default=Path("outputs/classical_baselines/xrdct_bundles"),
        help="Optional directory of calibrated 25/50 um XRD-CT .npz bundles to convert to OME-Zarr v0.4.",
    )
    args = parser.parse_args(argv)

    summary = build_derived_release_package(
        out_dir=args.out_dir,
        xrdct_bundles_dir=args.xrdct_bundles_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
