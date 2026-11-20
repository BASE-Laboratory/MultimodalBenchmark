"""Tests verifying MultimodalBenchmark's Layer 4 integration with basecoat v0.1."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import zarr
from basecoat import read_ome_zarr

from baselines.integrations import (
    load_benchmark_7dof_transform,
    load_benchmark_basecoat_transform,
)
from registration.i13_ladder import numpy_zyx_to_sitk, resample_moving_to_fixed_zyx
from release_packaging.build_derived_release import write_ome_zarr_v04


def test_ome_zarr_packager_roundtrips_via_basecoat_with_provenance(tmp_path: Path) -> None:
    vol = np.random.default_rng(123).normal(10.0, 1.5, size=(64, 64, 64)).astype(np.float32)
    store = write_ome_zarr_v04(
        vol,
        tmp_path / "calibrated_subvol.ome.zarr",
        voxel_size_um=(0.8006, 0.8006, 0.8006),
        multiscale_levels=2,
        extra_attrs={"track": "raw_float32", "scan": 169066},
    )

    grp = zarr.open_group(str(store), mode="r")
    assert grp.attrs["multiscales"][0]["name"] == "calibrated_subvol"

    lvl0 = read_ome_zarr(store, level=0)
    lvl1 = read_ome_zarr(store, level=1)
    np.testing.assert_allclose(lvl0.data, vol)
    assert np.allclose(lvl0.spacing_in("um"), (0.8006, 0.8006, 0.8006))
    assert lvl1.shape == (32, 32, 32)
    assert np.allclose(lvl1.spacing_in("um"), (1.6012, 1.6012, 1.6012))
    # Index-0 cell-centre shift under 2x block averaging: +0.5 * 0.8006 = 0.4003 um
    assert np.allclose(lvl1.origin_in("um"), (0.4003, 0.4003, 0.4003))
    assert "basecoat_version" in lvl0.provenance
    assert lvl0.provenance["zn13x_bench"]["scan"] == 169066


def test_frozen_7dof_basecoat_transform_matches_sitk_interior_resampling() -> None:

    hr_shape = (48, 48, 48)
    lr_shape = (32, 32, 32)
    hr_vox = 0.8125
    lr_vox = 1.625

    zz, yy, xx = np.indices(lr_shape, dtype=np.float32)
    lr_vol = (
        40.0
        + 15.0 * np.sin(0.22 * zz) * np.cos(0.19 * yy)
        + 10.0 * np.exp(-0.5 * ((zz - 14.0) ** 2 + (yy - 16.0) ** 2 + (xx - 15.0) ** 2) / 18.0)
    ).astype(np.float32)

    cx = 0.5 * (hr_shape[2] - 1) * hr_vox
    cy = 0.5 * (hr_shape[1] - 1) * hr_vox
    cz = 0.5 * (hr_shape[0] - 1) * hr_vox

    bc_tx, _ = load_benchmark_basecoat_transform("2p026x", center_xyz_um=(cx, cy, cz))
    dto_tx, _ = load_benchmark_7dof_transform("2p026x", center_xyz_um=(cx, cy, cz))
    assert bc_tx.from_space == "i13_169066"
    assert bc_tx.to_space == "i13_169067"

    res_bc = resample_moving_to_fixed_zyx(
        lr_vol,
        (lr_vox, lr_vox, lr_vox),
        hr_shape,
        (hr_vox, hr_vox, hr_vox),
        bc_tx,
    )

    mov_img = numpy_zyx_to_sitk(lr_vol, (lr_vox, lr_vox, lr_vox))
    sitk_f = sitk.ResampleImageFilter()
    sitk_f.SetSize((hr_shape[2], hr_shape[1], hr_shape[0]))
    sitk_f.SetOutputSpacing((hr_vox, hr_vox, hr_vox))
    sitk_f.SetOutputOrigin((0.0, 0.0, 0.0))
    sitk_f.SetTransform(dto_tx.to_sitk())
    sitk_f.SetInterpolator(sitk.sitkLinear)
    sitk_f.SetDefaultPixelValue(float(np.percentile(lr_vol, 5.0)))
    res_sitk = sitk.GetArrayFromImage(sitk_f.Execute(mov_img)).astype(np.float32)

    # Interior sub-cube strictly inside [0, N_lr - 1]^3 matches SimpleITK to float32 precision
    np.testing.assert_allclose(res_bc[8:40, 8:40, 8:40], res_sitk[8:40, 8:40, 8:40], atol=1e-5)

    # Verify nearest (1) and BSpline (23) enum constants map cleanly and bogus values raise
    res_nn = resample_moving_to_fixed_zyx(
        lr_vol, (lr_vox,) * 3, (16, 16, 16), (hr_vox,) * 3, bc_tx, interpolator=sitk.sitkNearestNeighbor
    )
    res_bs = resample_moving_to_fixed_zyx(
        lr_vol, (lr_vox,) * 3, (16, 16, 16), (hr_vox,) * 3, bc_tx, interpolator=sitk.sitkBSpline
    )
    assert res_nn.shape == res_bs.shape == (16, 16, 16)
    with pytest.raises(ValueError, match="Unsupported interpolator"):
        resample_moving_to_fixed_zyx(lr_vol, (lr_vox,) * 3, (8, 8, 8), (hr_vox,) * 3, bc_tx, interpolator=999)


def test_no_direct_zarr_or_numcodecs_imports_outside_tests() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prod_dirs = ("src", "registration", "preprocessing", "tasks", "splits", "baselines", "release_packaging", "scripts")
    offenders = []
    for sub in prod_dirs:
        d = repo_root / sub
        if not d.exists():
            continue
        for py in d.rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in ("zarr", "numcodecs"):
                            offenders.append(f"{py.relative_to(repo_root)}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] in ("zarr", "numcodecs"):
                        offenders.append(f"{py.relative_to(repo_root)}:{node.lineno}")
    assert not offenders, f"Direct zarr/numcodecs imports found in production modules (delegate to basecoat): {offenders}"


def test_evaluator_scores_ome_zarr_data_contract_stores_directly(tmp_path: Path) -> None:
    from basecoat import Volume, write_ome_zarr
    from multimodal_benchmark.evaluator import evaluate_bundles, load_bundle

    rng = np.random.default_rng(7)
    ref_arr = (50.0 + 20.0 * rng.random((24, 24, 24))).astype(np.float32)
    pred_arr = (ref_arr + rng.normal(0.0, 1.0, size=ref_arr.shape)).astype(np.float32)

    ref_store = write_ome_zarr(
        Volume.from_array(ref_arr, voxel_size_um=0.8006, provenance={"zn13x_bench": {"reference_track": "raw_float32"}}),
        tmp_path / "ref_t1.ome.zarr",
    )
    pred_store = write_ome_zarr(
        Volume.from_array(pred_arr, voxel_size_um=0.8006),
        tmp_path / "pred_t1.ome.zarr",
    )

    ref_b = load_bundle(ref_store)
    pred_b = load_bundle(pred_store)
    assert np.isclose(ref_b["_calibrated_voxel_size_um"], 0.8006)
    res = evaluate_bundles("t1", pred_b, ref_b, method_name="OMEZarrTool", with_transport=False, bootstrap_samples=20)
    assert res["metrics"]["psnr_db"] > 20.0
    assert res["reference_track"] == "raw_float32"


def test_extract_benchmark_tracks_uses_basecoat_cell_centre_coarsening() -> None:
    from preprocessing.extract_benchmark_tracks import build_ome_ngff_multiscales_attrs

    ngff = build_ome_ngff_multiscales_attrs("zn13x_t1_2p026x", voxel_size_um=0.8006)
    lite_tfms = ngff["multiscales"][0]["datasets"][1]["coordinateTransformations"]
    assert lite_tfms[0] == {"type": "scale", "scale": [4.0 * 0.8006] * 3}
    # 4x block averaging shifts index-0 cell centre by 0.5 * (4 - 1) * v = 1.5 * v
    assert np.allclose(lite_tfms[1]["translation"], [1.5 * 0.8006] * 3)
