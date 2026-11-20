"""Tests for leakage-free 3D splits, classical baselines, OpenImpala transport metrics, and evaluate.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import downscale_local_mean

from baselines.classical import (
    bicubic_chemical_upsample,
    cnmf_coupled_fusion,
    fbp_sparse_reconstruct_2d,
    guided_filter_chemical_upsample,
    guided_tv_chemical_fusion,
    joint_bilateral_upsample,
    lanczos_upsample_3d,
    multi_otsu_watershed_3d,
    sirt_reconstruct_2d,
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

    # Check LR-voxel axial guard bands (patch_z_lr + receptive_field_lr) above ruby shadow
    lr_bufs = payload["lr_voxel_buffers"]
    assert lr_bufs["t1_2p026x_169067_to_169066"]["buffer_lr_vox"] == 48
    assert lr_bufs["t1_8p101x_169068_to_169065"]["buffer_lr_vox"] == 40
    assert min(b["z0"] for b in payload["t1_0p8125um_hr_blocks"]) >= 640

    for key in ("t1_0p8125um_hr_blocks", "t1_0p325um_hr_blocks", "t2_t4_diad_0p54um_blocks"):
        boxes = [BoundingBox3D(**b) for b in payload[key]]
        ok, min_sep = verify_no_spatial_leakage(boxes, min_guard_band_um=32.0)
        assert ok is True
        assert min_sep >= 32.0

    # Check 4-fold Leave-One-Slice-Out (LOSO) CV for Task T2
    t2_loso = payload["t2_chemical_fusion_loso"]
    assert t2_loso["evaluation_protocol"] == "leave_one_slice_out_4fold_cv"
    assert len(t2_loso["loso_folds"]) == 4
    for fold in t2_loso["loso_folds"].values():
        held = fold["test_25um_scan"]
        train_scans = fold["train_25um_scans"]
        assert held not in train_scans
        assert len(train_scans) == 3
        allowed_50 = set(fold["allowed_self_supervised_50um_scans"])
        excl_50 = set(fold["excluded_guard_band_50um_scans"])
        assert allowed_50.isdisjoint(excl_50)


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
    assert res_tri["reference_track"] == "raw_float32"
    assert m["psnr_db"] > 18.0
    assert m["fringe_masked_psnr_db"] > 18.0
    assert m["bandlimited_psnr_db"] >= m["psnr_db"]
    assert 0.0 < m["fsc_0p5_cyc_per_vox"] <= 0.5
    assert m["d_eff_rel_err"] < 0.35
    assert "threshold_sens_d_eff_rel_err_span" in m
    assert np.isfinite(m["threshold_sens_d_eff_rel_err_span"])
    assert np.isfinite(m["tortuosity_rel_err"])
    md = format_markdown_leaderboard_row(res_tri)
    assert "| Tricubic | T1 |" in md


def test_t2_guided_fusion_baselines_beat_unguided_bicubic() -> None:
    """Verify Guided Filter, JBU, Guided-TV, and CNMF outperform unguided Bicubic on Task T2 chemical fusion."""
    ny, nx = 40, 40
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - 20.0, xx - 20.0)
    pellet = (rr <= 15.0).astype(np.float32)
    # Sharp internal macropores where chemical abundance MUST be zero
    pores = (((yy - 17.0) ** 2 + (xx - 17.0) ** 2) <= 3.2 ** 2) | (((yy - 24.0) ** 2 + (xx - 23.0) ** 2) <= 3.0 ** 2)
    solid = pellet * (~pores)

    # Radial core-shell Zn/Na exchange front around r = 9.5 px
    zn_frac = 1.0 / (1.0 + np.exp(-(rr - 9.5) / 1.5))
    zn_gt = (solid * (2.0 + 8.0 * zn_frac)).astype(np.float32)
    na_gt = (solid * (2.0 + 8.0 * (1.0 - zn_frac))).astype(np.float32)
    xct_guide = solid * 1.5 + 0.05

    zn_50 = downscale_local_mean(zn_gt, (2, 2)).astype(np.float32)
    na_50 = downscale_local_mean(na_gt, (2, 2)).astype(np.float32)
    sigma2 = np.full((ny, nx), 0.5, dtype=np.float32)

    ref_bundle = {
        "zn": zn_gt,
        "na": na_gt,
        "zn_sigma2": sigma2,
        "na_sigma2": sigma2,
        "pellet_mask": (solid > 0).astype(np.uint8),
        "hr_pore_mask": (solid == 0).astype(np.uint8),
        "hr_solid_mask": (solid > 0).astype(np.uint8),
    }

    bic_zn = bicubic_chemical_upsample(zn_50, (ny, nx))
    bic_na = bicubic_chemical_upsample(na_50, (ny, nx))
    res_bic = evaluate_bundles("t2", {"zn": bic_zn, "na": bic_na, "hr_zn": bic_zn, "hr_na": bic_na}, ref_bundle, "Bicubic")

    pm = solid == 0
    gf_zn = guided_filter_chemical_upsample(zn_50, xct_guide, pore_mask_yx=pm)
    gf_na = guided_filter_chemical_upsample(na_50, xct_guide, pore_mask_yx=pm)
    res_gf = evaluate_bundles("t2", {"zn": gf_zn, "na": gf_na, "hr_zn": gf_zn, "hr_na": gf_na}, ref_bundle, "GuidedFilter")

    gtv_zn = guided_tv_chemical_fusion(zn_50, xct_guide, pore_mask_yx=pm)
    gtv_na = guided_tv_chemical_fusion(na_50, xct_guide, pore_mask_yx=pm)
    res_gtv = evaluate_bundles("t2", {"zn": gtv_zn, "na": gtv_na, "hr_zn": gtv_zn, "hr_na": gtv_na}, ref_bundle, "Guided-TV")

    jbu_zn = joint_bilateral_upsample(zn_50, xct_guide, pore_mask_yx=pm)
    jbu_na = joint_bilateral_upsample(na_50, xct_guide, pore_mask_yx=pm)
    res_jbu = evaluate_bundles("t2", {"zn": jbu_zn, "na": jbu_na}, ref_bundle, "JBU")

    cnmf_zn, cnmf_na = cnmf_coupled_fusion(zn_50, na_50, xct_guide, pore_mask_yx=pm)
    res_cnmf = evaluate_bundles("t2", {"zn": cnmf_zn, "na": cnmf_na}, ref_bundle, "CNMF")

    assert res_gf["metrics"]["mean_rmse"] < res_bic["metrics"]["mean_rmse"]
    assert res_gtv["metrics"]["mean_rmse"] < res_bic["metrics"]["mean_rmse"]
    assert res_gtv["metrics"]["mean_bandlimited_rmse"] < res_bic["metrics"]["mean_bandlimited_rmse"]
    assert res_jbu["metrics"]["mean_rmse"] < res_bic["metrics"]["mean_rmse"]
    assert res_cnmf["metrics"]["mean_rmse"] < res_bic["metrics"]["mean_rmse"]
    assert np.isfinite(res_gf["metrics"]["in_pellet_na_zn_corr_abs_err"])
    assert np.isfinite(res_gf["metrics"]["reaction_front_pos_err_um"])
    assert res_gtv["metrics"]["t3b_pore_leakage_rel"] < 1e-6

    # Verify Nyquist checkerboard streak noise (k = 0.707 > 0.43 cyc/px) does NOT penalize band-limited T2 RMSE!
    checker = (((yy + xx) % 2) * 2 - 1).astype(np.float32) * 1.5
    ref_noisy = {**ref_bundle, "zn": zn_gt + checker, "na": na_gt - checker}
    res_clean_pred_vs_noisy_ref = evaluate_bundles("t2", {"zn": zn_gt, "na": na_gt}, ref_noisy, "OracleClean")
    assert res_clean_pred_vs_noisy_ref["metrics"]["mean_bandlimited_rmse"] < 1e-4
    assert res_clean_pred_vs_noisy_ref["metrics"]["mean_rmse"] > 1.4


def test_t3_cross_beamline_registration_and_t4_sparse_view_sirt() -> None:
    """Verify Task T3 cross-beamline registration metrics and Optional Task T4 sparse-view FBP vs SIRT."""
    rng = np.random.default_rng(7)
    fiducials = rng.uniform(-400.0, 400.0, size=(12, 3)).astype(np.float64)
    octants = rng.uniform(-250.0, 250.0, size=(8, 3)).astype(np.float64)
    s_ref = 0.54 / 0.8006
    t_ref = np.array([12.5, -8.0, 4.2], dtype=np.float64)
    ref_mapped_fid = fiducials * s_ref + t_ref
    ref_mapped_oct = octants * s_ref + t_ref

    # Slight translation + 0.1% scale perturbation
    s_pred = s_ref * 1.001
    t_pred = t_ref + np.array([0.5, -0.5, 0.3], dtype=np.float64)
    pred_reg = {
        "mapped_fiducials_xyz_um": fiducials * s_pred + t_pred,
        "mapped_octants_xyz_um": octants * s_pred + t_pred,
        "scale": np.array(s_pred),
    }
    ref_reg = {
        "fiducial_points_xyz_um": fiducials,
        "mapped_fiducials_xyz_um": ref_mapped_fid,
        "pellet_octants_xyz_um": octants,
        "mapped_octants_xyz_um": ref_mapped_oct,
        "scale": np.array(s_ref),
    }
    res_t3 = evaluate_bundles("t3", pred_reg, ref_reg, method_name="7DOF-NMI")
    assert res_t3["task"] == "t3"
    assert res_t3["metrics"]["fiducial_tre_mean_um"] < 1.5
    assert abs(res_t3["metrics"]["scale_rel_err_pct"] - 0.1) < 1e-3
    assert res_t3["metrics"]["octant_residual_drift_mean_um"] < 1.5

    # Optional Task T4: Sparse-view XRD-CT sinogram reconstruction (16 views vs 64-view reference)
    from skimage.transform import radon

    n = 32
    yy, xx = np.ogrid[:n, :n]
    phantom = (((yy - 16.0) ** 2 + (xx - 16.0) ** 2) <= 10.0 ** 2).astype(np.float32) * 5.0
    phantom[((yy - 14.0) ** 2 + (xx - 18.0) ** 2) <= 3.0 ** 2] = 0.0

    angles_full = np.linspace(0.0, 180.0, 64, endpoint=False)
    sino_full = radon(phantom, theta=angles_full, circle=True).T.astype(np.float32)
    idx_sub = np.arange(0, 64, 4)
    sino_sub = sino_full[idx_sub, :]
    angles_sub = angles_full[idx_sub]

    pred_fbp = fbp_sparse_reconstruct_2d(sino_sub, angles_sub)
    pred_sirt = sirt_reconstruct_2d(sino_sub, angles_sub, n_iter=45)

    res_fbp = evaluate_bundles("t4", {"recon": pred_fbp}, {"recon": phantom}, method_name="FBP-16view")
    res_sirt = evaluate_bundles("t4", {"recon": pred_sirt}, {"recon": phantom}, method_name="SIRT-16view")
    assert res_sirt["task"] == "t4"
    assert res_sirt["metrics"]["rmse"] < res_fbp["metrics"]["rmse"]


def test_legacy_multi_otsu_segmentation_helper() -> None:
    rng = np.random.default_rng(99)
    gt = np.zeros((20, 20, 20), dtype=np.int32)
    gt[:, 4:16, 4:16] = 1
    gt[6:14, 8:14, 8:14] = 2
    gt[:, 9:11, 9:11] = 0

    xct = (gt.astype(np.float32) * 3.0) + rng.normal(0.0, 0.2, size=gt.shape).astype(np.float32)
    pred_lbl = multi_otsu_watershed_3d(xct, num_classes=3)

    res = evaluate_bundles("t4a", {"labels": pred_lbl}, {"labels": gt}, method_name="Multi-Otsu")
    assert res["metrics"]["macro_dice"] > 0.85
    assert res["metrics"]["mean_phase_fraction_mae"] < 0.05

