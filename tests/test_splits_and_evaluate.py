"""Tests for leakage-free 3D splits, classical baselines, OpenImpala transport metrics, and evaluate.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import downscale_local_mean

from baselines.classical import (
    bicubic_chemical_upsample,
    cnmf_coupled_fusion,
    guided_tv_chemical_fusion,
    joint_bilateral_upsample,
    lanczos_upsample_3d,
    multi_otsu_watershed_3d,
    tricubic_upsample_3d,
)
from evaluate import evaluate_bundles, format_markdown_leaderboard_row, main as evaluate_cli_main
from splits.generate_splits import (
    BoundingBox3D,
    generate_benchmark_splits,
    verify_no_spatial_leakage,
    verify_subvolume_percolation,
)


def test_splits_have_zero_spatial_and_axial_leakage(tmp_path: Path) -> None:
    out_json = tmp_path / "splits.json"
    payload = generate_benchmark_splits(out_json)
    assert out_json.exists()

    for key in ("t1_0p8125um_hr_blocks", "t1_0p325um_hr_blocks", "t2_t4_diad_0p54um_blocks"):
        boxes = [BoundingBox3D(**b) for b in payload[key]]
        ok, min_sep = verify_no_spatial_leakage(boxes, min_guard_band_um=32.0)
        assert ok is True
        assert min_sep >= 32.0

    t3 = payload["t3_correlative_fusion"]
    allowed = set(t3["train_self_supervised_50um_scans"])
    excluded = set(t3["excluded_guard_band_50um_scans"])
    assert allowed.isdisjoint(excluded)
    assert len(excluded) >= 12


def test_t1_classical_baselines_and_openimpala_transport(tmp_path: Path) -> None:
    """Create a 3-axis percolating porous 3D phantom, run Tricubic & Lanczos 2x SR, and score via evaluate.py."""
    n = 24
    hr = np.ones((n, n, n), dtype=np.float32) * 2.0
    # Carve orthogonal percolating pore channels (dark = 0.2)
    hr[:, 8:16, 8:16] = 0.2
    hr[8:16, :, 8:16] = 0.2
    hr[8:16, 8:16, :] = 0.2
    hr = gaussian_filter(hr, sigma=0.6)

    perc = verify_subvolume_percolation(hr < 1.1)
    assert perc["certified_rev"] is True

    lr = downscale_local_mean(hr, (2, 2, 2)).astype(np.float32)
    pred_tri = tricubic_upsample_3d(lr, hr.shape)
    pred_lan = lanczos_upsample_3d(lr, hr.shape)

    pred_path = tmp_path / "t1_pred.npz"
    ref_path = tmp_path / "t1_ref.npz"
    out_json = tmp_path / "t1_score.json"
    np.savez_compressed(pred_path, pred=pred_lan)
    np.savez_compressed(ref_path, pred=hr)

    rc = evaluate_cli_main(
        [
            "--task", "t1",
            "--prediction", str(pred_path),
            "--reference", str(ref_path),
            "--method", "Lanczos-3D",
            "--out-json", str(out_json),
            "--markdown",
        ]
    )
    assert rc == 0
    assert out_json.exists()

    res_tri = evaluate_bundles("t1", {"pred": pred_tri}, {"pred": hr}, method_name="Tricubic")
    m = res_tri["metrics"]
    assert m["psnr_db"] > 18.0
    assert 0.0 < m["fsc_0p5_cyc_per_vox"] <= 0.5
    assert m["d_eff_rel_err"] < 0.35
    assert np.isfinite(m["tortuosity_rel_err"])
    md = format_markdown_leaderboard_row(res_tri)
    assert "| Tricubic | T1 |" in md


def test_t3_guided_fusion_baselines_beat_unguided_bicubic() -> None:
    """Verify structure-guided JBU, Guided-TV, and CNMF outperform unguided Bicubic on a sharp porous pellet slice."""
    ny, nx = 40, 40
    yy, xx = np.ogrid[:ny, :nx]
    pellet = (((yy - 20.0) ** 2 + (xx - 20.0) ** 2) <= 15.0 ** 2).astype(np.float32)
    # Sharp internal macropores where chemical abundance MUST be zero
    pores = (((yy - 17.0) ** 2 + (xx - 17.0) ** 2) <= 3.2 ** 2) | (((yy - 24.0) ** 2 + (xx - 23.0) ** 2) <= 3.0 ** 2)
    solid = pellet * (~pores)

    zn_gt = solid * ((xx >= 20).astype(np.float32) * 10.0 + (xx < 20).astype(np.float32) * 2.0)
    na_gt = solid * ((xx < 20).astype(np.float32) * 9.0 + (xx >= 20).astype(np.float32) * 2.0)
    xct_guide = solid * 1.5 + 0.05

    zn_50 = downscale_local_mean(zn_gt, (2, 2)).astype(np.float32)
    na_50 = downscale_local_mean(na_gt, (2, 2)).astype(np.float32)
    sigma2 = np.full((ny, nx), 0.5, dtype=np.float32)

    ref_bundle = {
        "zn": zn_gt,
        "na": na_gt,
        "zn_sigma2": sigma2,
        "na_sigma2": sigma2,
        "hr_pore_mask": (solid == 0).astype(np.uint8),
        "hr_solid_mask": (solid > 0).astype(np.uint8),
    }

    bic_zn = bicubic_chemical_upsample(zn_50, (ny, nx))
    bic_na = bicubic_chemical_upsample(na_50, (ny, nx))
    res_bic = evaluate_bundles("t3", {"zn": bic_zn, "na": bic_na, "hr_zn": bic_zn, "hr_na": bic_na}, ref_bundle, "Bicubic")

    pm = solid == 0
    gtv_zn = guided_tv_chemical_fusion(zn_50, xct_guide, pore_mask_yx=pm)
    gtv_na = guided_tv_chemical_fusion(na_50, xct_guide, pore_mask_yx=pm)
    res_gtv = evaluate_bundles("t3", {"zn": gtv_zn, "na": gtv_na, "hr_zn": gtv_zn, "hr_na": gtv_na}, ref_bundle, "Guided-TV")

    jbu_zn = joint_bilateral_upsample(zn_50, xct_guide, pore_mask_yx=pm)
    jbu_na = joint_bilateral_upsample(na_50, xct_guide, pore_mask_yx=pm)
    res_jbu = evaluate_bundles("t3", {"zn": jbu_zn, "na": jbu_na}, ref_bundle, "JBU")

    cnmf_zn, cnmf_na = cnmf_coupled_fusion(zn_50, na_50, xct_guide, pore_mask_yx=pm)
    res_cnmf = evaluate_bundles("t3", {"zn": cnmf_zn, "na": cnmf_na}, ref_bundle, "CNMF")

    assert res_gtv["metrics"]["mean_rmse"] < res_bic["metrics"]["mean_rmse"]
    assert res_jbu["metrics"]["mean_rmse"] < res_bic["metrics"]["mean_rmse"]
    assert res_cnmf["metrics"]["mean_rmse"] < res_bic["metrics"]["mean_rmse"]
    assert res_gtv["metrics"]["t3b_pore_leakage_rel"] < 1e-6


def test_t4_multi_otsu_segmentation_and_evaluation() -> None:
    rng = np.random.default_rng(99)
    gt = np.zeros((20, 20, 20), dtype=np.int32)
    gt[:, 4:16, 4:16] = 1
    gt[6:14, 8:14, 8:14] = 2
    # Carve percolating pore tunnel (class 0)
    gt[:, 9:11, 9:11] = 0

    xct = (gt.astype(np.float32) * 3.0) + rng.normal(0.0, 0.2, size=gt.shape).astype(np.float32)
    pred_lbl = multi_otsu_watershed_3d(xct, num_classes=3)

    res = evaluate_bundles("t4a", {"labels": pred_lbl}, {"labels": gt}, method_name="Multi-Otsu")
    assert res["metrics"]["macro_dice"] > 0.85
    assert res["metrics"]["mean_phase_fraction_mae"] < 0.05
