"""Tests for Savu (X, Z, Y) -> (Z, Y, X) out-of-core slicing and X-FUSE / OpenLSR-X integration adapters."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from baselines.integrations import align_openlsrx_7dof_pair, load_xfuse_benchmark_t3_bundle
from multimodal_benchmark.io import get_xct_shape_and_dtype, load_xct_subvolume


def test_savu_xzy_out_of_core_transposition(tmp_path: Path) -> None:
    h5_path = tmp_path / "mock_savu.h5"
    # Savu layout: (X=20, Z_vertical=12, Y=20)
    raw_xzy = np.arange(20 * 12 * 20, dtype=np.float32).reshape(20, 12, 20)
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("4-AstraReconGpu-tomo/data", data=raw_xzy)

    shape_zyx, _, ds_name = get_xct_shape_and_dtype(h5_path)
    assert ds_name == "4-AstraReconGpu-tomo/data"
    assert shape_zyx == (12, 20, 20)

    sub = load_xct_subvolume(h5_path, zyx_slices=(slice(2, 6), slice(4, 10), slice(5, 13)))
    expected = np.transpose(raw_xzy[5:13, 2:6, 4:10], (1, 2, 0))
    np.testing.assert_allclose(sub, expected)


@pytest.mark.skipif(
    not Path("data/cache/zenodo/output_43324_zn.nxs").exists(),
    reason="Cached Zenodo XRD-CT outputs not present",
)
def test_xfuse_adapter_loads_real_zenodo_t3_bundle() -> None:
    bundle = load_xfuse_benchmark_t3_bundle("data/cache/zenodo")
    assert bundle["guide_50um_zyx"]["Na"].shape == (20, 20, 20)
    assert bundle["guide_50um_zyx"]["Zn"].shape == (20, 20, 20)
    assert set(bundle["held_out_test_25um"].keys()) == {43322, 43323, 43324, 43325}
    # Verify guard-band mask holds out the majority of adjacent 50 um slices
    assert 1 <= int(np.sum(bundle["train_self_sup_mask_50um"])) <= 10


def test_openlsrx_7dof_adapter_aligns_synthetic_pair() -> None:
    zz, yy, xx = np.indices((24, 32, 32), dtype=np.float32)
    hr = (
        np.exp(-0.5 * ((zz - 8) ** 2 + (yy - 10) ** 2 + (xx - 11) ** 2) / 6.0)
        + 1.5 * np.exp(-0.5 * ((zz - 16) ** 2 + (yy - 21) ** 2 + (xx - 19) ** 2) / 7.0)
        + 1.2 * np.exp(-0.5 * ((zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 22) ** 2) / 5.0)
    )
    lr = hr[::2, ::2, ::2].copy()
    lr_on_hr, hr_out, meta = align_openlsrx_7dof_pair(lr, hr, lr_voxel_um=1.625, hr_voxel_um=0.8125)
    assert lr_on_hr.shape == hr_out.shape == (24, 32, 32)
    assert 0.94 < meta["transform"]["scale"] < 1.06


def test_openlsrx_frozen_benchmark_transforms_and_two_track_extractor(tmp_path: Path) -> None:
    """Verify OpenLSR-X consumes frozen benchmark transforms, synthetic degradations, and reversible two-track HDF5s."""
    import json
    from baselines.integrations import (
        load_benchmark_7dof_transform,
        synthesize_degraded_lr_from_hr,
    )
    from preprocessing.extract_benchmark_tracks import extract_paired_subvolume_tracks

    tfm_2x, meta_2x = load_benchmark_7dof_transform("2.026x")
    tfm_8x, meta_8x = load_benchmark_7dof_transform("8.101x")
    assert abs(meta_2x["true_magnification_ratio"] - 2.0258) < 1e-4
    assert abs(meta_8x["true_magnification_ratio"] - 8.1014) < 1e-4
    assert abs(tfm_2x.scale - 0.98727) < 1e-4

    hr = np.ones((16, 32, 32), dtype=np.float32) * 75.0
    hr[:, 8:24, 8:24] = 85.0
    syn_clean = synthesize_degraded_lr_from_hr(hr, true_scale_ratio=2.0258, with_noise=False)
    syn_noisy = synthesize_degraded_lr_from_hr(hr, true_scale_ratio=2.0258, with_noise=True, seed=7)
    assert syn_clean.shape == syn_noisy.shape
    assert float(np.std(syn_noisy - syn_clean)) > 0.05

    out_h5 = tmp_path / "tracks.h5"
    rep = extract_paired_subvolume_tracks(
        lr_zyx=syn_noisy,
        hr_zyx=hr,
        lr_nominal_voxel_um=1.625,
        hr_nominal_voxel_um=0.8125,
        ruby_calibrated_hr_voxel_um=0.8006,
        track_name="2p026x",
        out_h5_path=out_h5,
    )
    assert rep["lite_4x_shape_zyx"] == [4, 8, 8]
    with h5py.File(out_h5, "r") as fh:
        raw_hr = fh["primary_raw_float32/hr"][:]
        cln_hr = fh["cleaned_dering_v1/hr"][:]
        diff_hr = fh["cleaned_dering_v1/diff_hr_rings"][:]
        np.testing.assert_allclose(cln_hr + diff_hr, raw_hr, atol=1e-5)
        ngff = json.loads(fh.attrs["ome_ngff_zattrs_json"])
        assert ngff["multiscales"][0]["version"] == "0.4"

    croissant = json.loads(Path("configs/croissant.json").read_text(encoding="utf-8"))
    assert croissant["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert len(croissant["creator"]) == 8
    assert len(croissant["recordSet"]) == 4

    # Verify registration-jitter synthetic control (`synthetic_jittered`) perturbs alignment deterministically
    syn_jit = synthesize_degraded_lr_from_hr(
        hr, true_scale_ratio=2.0258, with_noise=True, registration_jitter_hr_vox=2.5, seed=7
    )
    assert syn_jit.shape == syn_noisy.shape
    assert float(np.max(np.abs(syn_jit - syn_noisy))) > 0.01


def test_learned_cli_commands_match_upstream_openlsrx_and_xfuse_parsers(tmp_path: Path) -> None:
    """Ensure baselines/learned.py emits ONLY real upstream openlsrx and xfuse CLI subcommands and BibTeX DOIs."""
    import re
    import yaml
    from baselines.learned import build_openlsrx_commands, build_xfuse_commands

    t1_cmds = build_openlsrx_commands("configs/baselines/openlsrx_t1.yaml", tmp_path / "t1")
    # Default excludes exploratory 8x: 1 track (2p026x) * 4 regimes * 3 models * 3 seeds = 36 runs
    assert len(t1_cmds) == 36
    for c in t1_cmds:
        assert "train_and_predict" not in c
        assert "openlsrx fit --config" in c
        assert "openlsrx predict-volume" in c
        assert "pred_test.ome.zarr" in c
        assert "--voxel-size-um 0.8006" in c
        assert "real_paired_test_hr.ome.zarr" in c
        assert any(f"--model.backbone {b}" in c for b in ("srgan", "rrdb", "swinir"))

    t1_cmds_all = build_openlsrx_commands(
        "configs/baselines/openlsrx_t1.yaml", tmp_path / "t1_all", include_exploratory_8x=True
    )
    assert len(t1_cmds_all) == 72

    t2_cmds = build_xfuse_commands("configs/baselines/xfuse_t2.yaml", tmp_path / "t2")
    assert len(t2_cmds) == 36
    for c in t2_cmds:
        assert "fit_correlative" not in c
        assert ("xfuse fuse " in c) or ("xfuse fuse-3d " in c)

    cert = yaml.safe_load(Path("configs/i13_certified_regions.yaml").read_text(encoding="utf-8"))
    assert "zenodo_14245751_preview_8bit" in cert
    assert "169067_from_169066" in cert["zenodo_14245751_preview_8bit"]["adjacent_rungs"]

    bib_text = Path("paper/references.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{([^,]+),", bib_text)
    dois = re.findall(r"\bdoi\s*=\s*\{([^}]+)\}", bib_text)
    assert len(entries) == len(dois) >= 18
    assert "Christensen" not in bib_text


