"""Tests verifying MultimodalBenchmark's Layer 4 integration with basecoat v0.1."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

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


def test_load_bundle_unpacks_volumeset_uncertainties_and_ome_zarr_adapters(tmp_path: Path) -> None:
    from basecoat import Volume, VolumeSet, write_ome_zarr
    from baselines.integrations import (
        export_openlsrx_pair_ome_zarr,
        export_xfuse_t2_fold_ome_zarr,
    )
    from multimodal_benchmark.evaluator import load_bundle

    rng = np.random.default_rng(19)
    lr = rng.uniform(20.0, 80.0, size=(8, 8, 8)).astype(np.float32)
    hr = rng.uniform(20.0, 80.0, size=(16, 16, 16)).astype(np.float32)
    lr_store, hr_store = export_openlsrx_pair_ome_zarr(lr, hr, tmp_path / "openlsrx_pair")
    assert lr_store.is_dir() and hr_store.is_dir()
    assert np.allclose(read_ome_zarr(hr_store).spacing_in("um"), (0.8006, 0.8006, 0.8006))

    xf_written = export_xfuse_t2_fold_ome_zarr(
        {
            "diad_xct_zyx": hr,
            "train_25um_slices": {
                43322: {
                    "zn": np.ones((12, 12), dtype=np.float32),
                    "na": np.ones((12, 12), dtype=np.float32) * 2.0,
                }
            },
        },
        tmp_path / "xfuse_fold",
    )
    assert Path(xf_written["hr_volume_store"]).is_dir()
    assert Path(xf_written["guide_slices"][43322]).is_dir()

    # Verify load_bundle unpacks VolumeSet.uncertainties sidecars alongside conformal labels
    img = Volume.from_array(hr, voxel_size_um=0.54)
    phase = Volume.from_array((hr > 50.0).astype(np.uint8), voxel_size_um=0.54)
    phase_lo = Volume.from_array((hr > 60.0).astype(np.uint8), voxel_size_um=0.54)
    phase_hi = Volume.from_array((hr > 40.0).astype(np.uint8), voxel_size_um=0.54)
    sigma_ortho = Volume.from_array(np.full_like(hr, 0.05), voxel_size_um=0.54)
    bundle_store = write_ome_zarr(
        VolumeSet(
            img,
            labels={"phase": phase, "phase_lower": phase_lo, "phase_upper": phase_hi},
            uncertainties={"sigma_ortho": sigma_ortho},
        ),
        tmp_path / "pred_with_unc.ome.zarr",
    )
    loaded = load_bundle(bundle_store)
    assert "sigma_ortho" in loaded
    assert np.allclose(loaded["sigma_ortho"], 0.05)
    assert "phase_lower" in loaded and "phase_upper" in loaded
    assert np.array_equal(loaded["labels"], phase.data)


def test_mechanics_discrepancy_and_conformal_envelopes_via_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    from basecoat import Volume, VolumeSet, write_ome_zarr
    from multimodal_benchmark.evaluator import evaluate_bundles, main as eval_cli_main
    from multimodal_benchmark.metrics import mechanics as mech_mod

    mech_mod._REF_MECHANICS_CACHE.clear()

    mock_sms = ModuleType("sms")

    def _fake_effective_modulus(
        vol: Volume,
        *,
        direction: str = "z",
        youngs_modulus: float = 10.0e9,
        poisson_ratio: float = 0.25,
        solid_phase: int = 1,
        rtol: float = 1e-5,
        max_iter: int = 4000,
        check_converged: bool = False,
    ) -> tuple[float, Any]:
        _ = (direction, poisson_ratio, rtol, max_iter, check_converged)
        # Assert calibrated SI metres spacing via VolumeProtocol!
        sp_m = vol.spacing_in("m")
        assert len(sp_m) == 3 and all(s > 0.0 for s in sp_m)
        phi_s = float(np.mean(np.asarray(vol) == int(solid_phase)))
        return float(youngs_modulus) * (phi_s**2), SimpleNamespace(converged=True)

    mock_sms.effective_modulus = _fake_effective_modulus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sms", mock_sms)

    ref_lbl = np.zeros((16, 16, 16), dtype=np.uint8)
    ref_lbl[:, 3:13, 3:13] = 1
    pred_lbl = ref_lbl.copy()
    pred_lo = np.zeros_like(ref_lbl)
    pred_lo[:, 4:12, 4:12] = 1
    pred_hi = np.zeros_like(ref_lbl)
    pred_hi[:, 2:14, 2:14] = 1

    ref_store = write_ome_zarr(
        Volume.from_array(ref_lbl, voxel_size_um=0.54, label_names={0: "pore", 1: "solid"}),
        tmp_path / "ref_t4a.ome.zarr",
    )
    pred_store = write_ome_zarr(
        VolumeSet(
            Volume.from_array(pred_lbl.astype(np.float32), voxel_size_um=0.54),
            labels={
                "phase": Volume.from_array(pred_lbl, voxel_size_um=0.54),
                "phase_lower": Volume.from_array(pred_lo, voxel_size_um=0.54),
                "phase_upper": Volume.from_array(pred_hi, voxel_size_um=0.54),
            },
        ),
        tmp_path / "pred_t4a.ome.zarr",
    )

    out_json = tmp_path / "joint_score.json"
    rc = eval_cli_main(
        [
            "--task",
            "t4a",
            "--prediction",
            str(pred_store),
            "--reference",
            str(ref_store),
            "--with-mechanics",
            "--out-json",
            str(out_json),
        ]
    )
    assert rc == 0
    res = evaluate_bundles(
        "t4a",
        {
            "labels": pred_lbl,
            "phase_lower": pred_lo,
            "phase_upper": pred_hi,
            "_calibrated_voxel_size_um": 0.54,
        },
        {"labels": ref_lbl, "_calibrated_voxel_size_um": 0.54},
        with_transport=True,
        with_mechanics=True,
    )
    m = res["metrics"]
    assert m["e_eff_rel_err"] < 1e-6
    assert m["e_eff_conformal_lower_pa"] < m["e_eff_mean_ref_pa"] < m["e_eff_conformal_upper_pa"]
    assert m["e_eff_conformal_covers_ref"] == 1.0
    assert np.isfinite(m["tortuosity_rel_err"])


@pytest.mark.skipif(
    not __import__("multimodal_benchmark.metrics", fromlist=["is_sms_available"]).is_sms_available(),
    reason="Compiled Fortran SMS library (_libsms_core) not installed in current environment",
)
def test_golden_thread_live_openimpala_and_sms_on_same_ome_zarr_bundle(tmp_path: Path) -> None:
    """Golden Thread: score BOTH OpenImpala transport AND compiled Fortran SMS stiffness on one .ome.zarr bundle."""
    from basecoat import Volume, write_ome_zarr
    from multimodal_benchmark.evaluator import evaluate_bundles, load_bundle

    # 3-axis percolating porous scaffold (12^3) with continuous load-bearing solid struts AND open pore channels
    n = 12
    hr_ref = np.full((n, n, n), 85.0, dtype=np.float32)
    hr_ref[:, 4:8, 4:8] = 15.0
    hr_ref[4:8, :, 4:8] = 15.0
    hr_ref[4:8, 4:8, :] = 15.0

    hr_pred = hr_ref.copy()
    # Perturb two boundary voxels slightly
    hr_pred[3, 4, 4] = 85.0
    hr_pred[8, 7, 7] = 85.0

    ref_store = write_ome_zarr(
        Volume.from_array(hr_ref, voxel_size_um=0.8006),
        tmp_path / "golden_ref.ome.zarr",
    )
    pred_store = write_ome_zarr(
        Volume.from_array(hr_pred, voxel_size_um=0.8006),
        tmp_path / "golden_pred.ome.zarr",
    )

    res = evaluate_bundles(
        "t1",
        load_bundle(pred_store),
        load_bundle(ref_store),
        method_name="GoldenThreadCandidate",
        with_transport=True,
        with_mechanics=True,
        report_threshold_sensitivity=False,
        bootstrap_samples=0,
    )
    m = res["metrics"]
    assert m["percolates_all_axes_ref"] == 1.0
    assert np.isfinite(m["tortuosity_mean_ref"]) and m["tortuosity_mean_ref"] >= 1.0
    assert m["e_eff_mean_ref_pa"] > 0.0
    assert 0.0 < m["e_eff_norm_ref"] < 1.0
    assert m["e_eff_rel_err"] < 0.25
    assert m["d_eff_rel_err"] < 0.25
