"""Unit tests for Stage 0A-2 NeXus stage-z resolver, OME-Zarr v0.4 packager, Croissant 1.0 validator, and LaTeX aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import zarr

from registration.diad_correlative import read_dls_k11_raw_stage_z
from registration.diad_stage_z_audit import DEFAULT_SCANS_25UM, DEFAULT_SCANS_50UM, resolve_stage_z_matches
from release_packaging.build_derived_release import (
    build_derived_release_package,
    validate_croissant_jsonld,
    write_ome_zarr_v04,
)
from scripts.aggregate_results_to_latex import aggregate_score_jsons, format_booktabs_table


def test_stage0a2_nexus_stage_z_resolver_detects_exact_and_shifted_matches(tmp_path: Path) -> None:
    nexus_dir = tmp_path / "nexus"
    nexus_dir.mkdir()

    # Create synthetic raw DLS K11 NeXus files with 0.05 mm (50 um) vertical steps on entry/instrument/gts/gts_y
    z50_map = {}
    for idx, sid50 in enumerate(DEFAULT_SCANS_50UM):
        pos_mm = 10.0 + 0.05 * idx
        z50_map[idx] = pos_mm
        with h5py.File(nexus_dir / f"k11-{sid50}.nxs", "w") as fh:
            fh.create_dataset("entry/instrument/gts/gts_y", data=np.array([pos_mm], dtype=np.float64))

    exact_targets = {43324: z50_map[9], 43322: z50_map[14], 43323: z50_map[17], 43325: z50_map[18]}
    for sid25, pos_mm in exact_targets.items():
        with h5py.File(nexus_dir / f"k11-{sid25}.nxs", "w") as fh:
            fh.create_dataset("entry/instrument/gts/gts_y", data=np.array([pos_mm + 0.002], dtype=np.float64))

    raw_rep = read_dls_k11_raw_stage_z(nexus_dir, scan_ids=list(DEFAULT_SCANS_25UM) + list(DEFAULT_SCANS_50UM))
    res = resolve_stage_z_matches(raw_rep)
    assert res["status"] == "VERIFIED_ALL_AGREE"
    assert res["matches"]["43324"]["matched_50um_scan"] == 43345
    assert res["matches"]["43324"]["offset_to_best_um"] == 2.0

    # Now shift 43324 to z=10 (43346) and verify mismatch detection triggers cleanly
    with h5py.File(nexus_dir / "k11-43324.nxs", "w") as fh:
        fh.create_dataset("entry/instrument/gts/gts_y", data=np.array([z50_map[10]], dtype=np.float64))
    raw_rep2 = read_dls_k11_raw_stage_z(nexus_dir, scan_ids=list(DEFAULT_SCANS_25UM) + list(DEFAULT_SCANS_50UM))
    res2 = resolve_stage_z_matches(raw_rep2)
    assert res2["status"] == "MISMATCH_DETECTED_UPDATE_SPLITS"
    assert res2["matches"]["43324"]["matched_50um_scan"] == 43346
    assert res2["matches"]["43324"]["agrees_with_correlation_match"] is False


def test_croissant_validation_and_ome_zarr_v04_release_packager(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    chk = validate_croissant_jsonld(repo_root / "configs" / "croissant.json")
    assert chk["valid"] is True, f"Croissant errors: {chk['errors']}"
    assert chk["num_distributions"] >= 3
    assert chk["num_record_sets"] >= 3

    # Test 3D OME-Zarr v0.4 pyramid creation
    vol = np.random.default_rng(0).normal(1.0, 0.2, size=(64, 64, 64)).astype(np.float32)
    zstore = write_ome_zarr_v04(vol, tmp_path / "sample.ome.zarr", voxel_size_um=(0.8006, 0.8006, 0.8006), multiscale_levels=2)
    grp = zarr.open_group(str(zstore), mode="r")
    assert "0" in grp and "1" in grp
    ms = grp.attrs["multiscales"][0]
    assert ms["version"] == "0.4"
    assert ms["axes"][0]["unit"] == "micrometer"
    assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [0.8006, 0.8006, 0.8006]

    # Test full derived release package builder with a mock XRD-CT bundle
    xdir = tmp_path / "xrdct_bundles"
    xdir.mkdir()
    np.savez_compressed(
        xdir / "pair_25um_43322_50um_43395_z14.npz",
        scan_25um=np.int32(43322),
        scan_50um=np.int32(43395),
        z_index_50um=np.int32(14),
        zn_25um=np.ones((40, 40), dtype=np.float32),
        na_25um=np.ones((40, 40), dtype=np.float32),
        zn_sigma2=np.full((40, 40), 0.1, dtype=np.float32),
        na_sigma2=np.full((40, 40), 0.1, dtype=np.float32),
    )
    rel = build_derived_release_package(tmp_path / "release", repo_root=repo_root, xrdct_bundles_dir=xdir)
    assert rel["lite_under_2gb"] is True
    assert rel["num_ome_zarr_stores"] == 1
    assert (tmp_path / "release" / "metadata" / "zenodo_derived_metadata.json").exists()


def test_latex_booktabs_score_aggregator(tmp_path: Path) -> None:
    sdir = tmp_path / "scores"
    sdir.mkdir()
    for seed, psnr, deff in [(42, 31.2, 0.08), (137, 31.6, 0.06), (2026, 31.4, 0.07)]:
        (sdir / f"run_{seed}.json").write_text(
            json.dumps({"task": "t1", "method": "OpenLSR-X (real)", "metrics": {"psnr_db": psnr, "d_eff_rel_err": deff}}),
            encoding="utf-8",
        )
    agg = aggregate_score_jsons(sorted(sdir.glob("*.json")))
    tex = format_booktabs_table(agg, ["psnr_db", "d_eff_rel_err"])
    assert r"\toprule" in tex
    assert "OpenLSR-X (real)" in tex
    assert r"\pm" in tex
