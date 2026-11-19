"""Synthetic ground-truth tests for Stage 0A chirality/PSF alignment, Stage 0B 7-DOF ladder, and Stage 0C pellet slip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from registration.cross_beamline_audit import compute_phase_cnr, run_stage0c_pellet_vs_fiducial_audit
from registration.diad_correlative import align_xrdct_slice_to_diad_xct, blur_xct_to_xrdct_grid
from registration.i13_ladder import (
    SimilarityTransform3D,
    radial_non_cupping_mask,
    register_3d_similarity,
    resample_moving_to_fixed_zyx,
)


def test_diad_correlative_detects_astra_x_chirality_and_z_height() -> None:
    """Verify Stage 0A recovers flip_x=True, z-slice index, and sub-pixel shift on a 3D XCT + 2D XRD-CT phantom."""
    nz, ny, nx = 36, 80, 80
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    # Asymmetric 3D phantom whose z-profile varies cleanly with height
    xct = np.zeros((nz, ny, nx), dtype=np.float32)
    for iz in range(nz):
        r_pel = 18.0 + 0.25 * iz
        xct[iz] += 1.0 * (((yy[0] - 42.0) ** 2 + (xx[0] - 35.0) ** 2) <= r_pel ** 2)
        # Off-axis bright calibration bead that moves laterally with z to uniquely pin z & chirality
        bx = 55.0 - 0.35 * iz
        by = 24.0 + 0.20 * iz
        xct[iz] += 2.2 * (((yy[0] - by) ** 2 + (xx[0] - bx) ** 2) <= 6.5 ** 2)

    blurred = blur_xct_to_xrdct_grid(
        xct,
        xct_voxel_um=2.0,
        xrd_pixel_um=8.0,
        beam_thickness_um=8.0,
        target_yx_shape=(20, 20),
    )
    true_z = 21
    # Simulate XRD-CT reconstructed in flipped x chirality (gridrec vs AstraReconGpu) + shift
    from scipy.ndimage import shift as nd_shift

    xrd_slice = nd_shift(np.fliplr(blurred[true_z]), shift=(1.0, -1.0), order=1, mode="nearest")

    res = align_xrdct_slice_to_diad_xct(
        xrd_total_yx=xrd_slice,
        blurred_xct_zyx=blurred,
        xct_voxel_um=2.0,
        xrd_pixel_um=8.0,
        test_both_chiralities=True,
    )
    assert res.flip_x is True
    assert abs(res.best_z_center_vox - true_z) < 2.0
    assert res.pearson_r > 0.94


def test_i13_7dof_similarity_registration_and_cupping_mask() -> None:
    """Verify Stage 0B 3D 7-DOF similarity registration recovers translation, rotation, and scale."""
    shape = (40, 48, 48)
    spacing = (1.5, 1.5, 1.5)
    zz, yy, xx = np.indices(shape, dtype=np.float32)

    fixed = np.zeros(shape, dtype=np.float32)
    beads = [
        (14.0, 16.0, 15.0, 4.2, 2.0),
        (26.0, 31.0, 18.0, 4.8, 2.5),
        (19.0, 20.0, 33.0, 4.5, 2.2),
        (28.0, 15.0, 29.0, 3.8, 1.8),
    ]
    for cz, cy, cx, rad, val in beads:
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        fixed += val * np.exp(-0.5 * dist2 / ((rad * 0.5) ** 2))

    gt_sitk = sitk.Similarity3DTransform()
    gt_sitk.SetCenter((36.0, 36.0, 30.0))
    gt_sitk.SetRotation((0.3, -0.4, 0.8660254), 0.05)
    gt_sitk.SetTranslation((2.4, -1.8, 1.5))
    gt_sitk.SetScale(1.02)
    gt_dto = SimilarityTransform3D.from_sitk(gt_sitk.GetInverse())

    moving = resample_moving_to_fixed_zyx(fixed, spacing, shape, spacing, gt_dto)

    reg_res = register_3d_similarity(
        fixed_zyx=fixed,
        moving_zyx=moving,
        fixed_spacing_zyx_um=spacing,
        moving_spacing_zyx_um=spacing,
        metric="mean_squares",
        max_iterations=250,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        allow_scale=True,
    )
    assert abs(reg_res.transform.scale - 1.02) < 0.02
    assert reg_res.tre_mean_um < 1.5

    mask = radial_non_cupping_mask(shape, valid_radius_fraction=0.8)
    assert mask.shape == shape
    assert mask[20, 24, 24]
    assert not mask[20, 0, 0]


def test_stage0c_detects_intra_capillary_pellet_slip_and_exports_yaml(tmp_path: Path) -> None:
    """Simulate a pellet shifting inside the capillary relative to stationary fiducial beads."""
    shape = (36, 48, 48)
    spacing = (1.0, 1.0, 1.0)
    zz, yy, xx = np.indices(shape, dtype=np.float32)

    # Fiducials in upper/outer region; pellet in central region with rich internal structure
    fid_diad = (
        2.5 * np.exp(-0.5 * ((zz - 8) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) / 5.0)
        + 2.5 * np.exp(-0.5 * ((zz - 8) ** 2 + (yy - 38) ** 2 + (xx - 12) ** 2) / 5.0)
        + 2.5 * np.exp(-0.5 * ((zz - 28) ** 2 + (yy - 10) ** 2 + (xx - 38) ** 2) / 5.0)
    )
    pel_diad = (
        1.8 * np.exp(-0.5 * ((zz - 18) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) / 36.0)
        + 1.2 * np.exp(-0.5 * ((zz - 15) ** 2 + (yy - 21) ** 2 + (xx - 27) ** 2) / 6.0)
        + 1.4 * np.exp(-0.5 * ((zz - 22) ** 2 + (yy - 27) ** 2 + (xx - 21) ** 2) / 6.0)
    )
    diad = (fid_diad + pel_diad).astype(np.float32)

    # In I13-2 scan, fiducials stay unshifted, while pellet slipped by +3.5 um along X!
    from scipy.ndimage import shift as nd_shift

    pel_i13 = nd_shift(pel_diad, shift=(0.0, 0.0, 3.5), order=3, mode="nearest")
    # Also apply cross-beamline intensity scaling
    i13 = (1.7 * (fid_diad + pel_i13) + 0.4).astype(np.float32)

    fid_mask = (fid_diad > 0.25).astype(np.uint8)
    pel_mask = (pel_diad > 0.35).astype(np.uint8)
    zn_mask = ((pel_diad > 1.2) & (xx > 24)).astype(np.uint8)
    na_mask = ((pel_diad > 0.6) & (xx <= 24)).astype(np.uint8)

    qc_png = tmp_path / "stage0c_qc.png"
    report = run_stage0c_pellet_vs_fiducial_audit(
        diad_zyx=diad,
        i13_zyx=i13,
        diad_spacing_zyx_um=spacing,
        i13_spacing_zyx_um=spacing,
        diad_fiducial_mask_zyx=fid_mask,
        diad_pellet_mask_zyx=pel_mask,
        slip_threshold_um=1.5,
        octant_cert_threshold_um=1.5,
        min_octant_voxels=20,
        zn_mask_zyx=zn_mask,
        na_mask_zyx=na_mask,
        qc_png_path=qc_png,
    )

    assert report.pellet_moved_inside_capillary is True
    assert report.pellet_vs_fiducial_slip_um > 1.8
    assert report.certified_octant_count >= 1
    assert qc_png.exists()

    cert_yaml = tmp_path / "certified_regions.yaml"
    report.save_certified_regions_yaml(cert_yaml)
    loaded = yaml.safe_load(cert_yaml.read_text())
    assert loaded["pellet_moved_inside_capillary"] is True
    assert compute_phase_cnr(diad, zn_mask, na_mask) >= 0.0


def test_fit_calibration_sphere_3d_recovers_scale_on_border_clipped_sphere() -> None:
    """Verify 3D radial-normal Cauchy sphere fitter recovers sub-voxel radius even when clipped by a crop border."""
    from scipy.ndimage import gaussian_filter

    from registration.i13_ladder import fit_calibration_sphere_3d

    shape = (80, 96, 96)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    # Place sphere near y=86 so y > 95 is clipped by the border, and add an adjacent non-radial wall at x > 68
    cz, cy, cx, true_r_vox = 40.25, 76.4, 38.6, 26.5
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    vol = 60.0 + 35.0 * (dist <= true_r_vox).astype(np.float32)
    vol[:, :, 70:75] = 92.0  # distracting flat wall
    vol = gaussian_filter(vol, sigma=1.1)

    res = fit_calibration_sphere_3d(
        vol,
        nominal_voxel_um=2.0,
        seed_center_zyx_vox=(40.0, 76.0, 39.0),
        reference_diameter_um=104.0,  # true voxel = 104 / (2 * 26.5) = 1.96226 um (-1.89%)
        border_margin_vox=6,
        smooth_sigma_vox=1.0,
    )
    assert abs(res.radius_vox - true_r_vox) < 0.35
    assert res.sphericity_rmse_vox < 0.35
    assert abs(res.calibrated_voxel_um - (104.0 / (2.0 * true_r_vox))) < 0.03


def test_savu_hdf5_cli_entrypoints_for_scarf_and_hamilton(tmp_path: Path) -> None:
    """Verify diad_correlative.main and cross_beamline_audit.main stream Savu (X, Z, Y) HDF5 files cleanly."""
    import h5py

    from registration.cross_beamline_audit import main as stage0c_main
    from registration.diad_correlative import main as stage0a_main, read_dls_k11_raw_stage_z

    # Build synthetic (Z=32, Y=40, X=40) volume and save in Savu (X, Z, Y) = (40, 32, 40) layout!
    zz, yy, xx = np.indices((32, 40, 40), dtype=np.float32)
    pel = (15.0 * np.exp(-0.5 * ((zz - 16) ** 2 + (yy - 20) ** 2 + (xx - 22) ** 2) / 25.0)).astype(np.float32)
    # Add textured pores inside pellet + homogeneous sphere fiducial
    pores = (4.0 * np.sin(zz * 1.3) * np.cos(yy * 1.1) * (pel > 4.0)).astype(np.float32)
    fid = (22.0 * (((zz - 9) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) <= 25.0)).astype(np.float32)
    vol_zyx = 50.0 + pel + pores + fid

    diad_h5 = tmp_path / "tomo_p4_astra_recon_gpu.h5"
    i13_h5 = tmp_path / "169067_recon.h5"
    # Savu layout stores (X, Z, Y) so np.transpose(raw, (1, 2, 0)) -> (Z, Y, X)
    savu_xzy = np.transpose(vol_zyx, (2, 0, 1))
    with h5py.File(diad_h5, "w") as f:
        f.create_dataset("4-AstraReconGpu-tomo/data", data=savu_xzy)
    with h5py.File(i13_h5, "w") as f:
        f.create_dataset("4-TomopyRecon-tomo/data", data=savu_xzy)

    # Synthetic raw DLS K11 NeXus file with /entry/diffraction/gts_y
    raw_nxs_dir = tmp_path / "dls_nexus"
    raw_nxs_dir.mkdir()
    with h5py.File(raw_nxs_dir / "k11-43324.nxs", "w") as f:
        f.create_dataset("entry/diffraction/gts_y", data=np.array([12.45]))
    z_meta = read_dls_k11_raw_stage_z(raw_nxs_dir, scan_ids=(43324,))
    assert z_meta["available"] is True
    assert abs(z_meta["scans"]["43324"]["entry/diffraction/gts_y"] - 12.45) < 1e-6

    out_json = tmp_path / "correlative.json"
    rc_a = stage0a_main(
        [
            "--diad-xct-h5",
            str(diad_h5),
            "--xrdct-dir",
            str(tmp_path),
            "--dls-raw-nexus-dir",
            str(raw_nxs_dir),
            "--z-stride",
            "1",
            "--yx-stride",
            "1",
            "--out-json",
            str(out_json),
        ]
    )
    assert rc_a == 0 and out_json.exists()

    out_yaml = tmp_path / "stage0c.yaml"
    qc_png = tmp_path / "stage0c_qc.png"
    rc_c = stage0c_main(
        [
            "--i13-h5",
            str(i13_h5),
            "--diad-h5",
            str(diad_h5),
            "--i13-voxel-um",
            "1.0",
            "--diad-voxel-um",
            "1.0",
            "--i13-stride",
            "1",
            "--diad-stride",
            "1",
            "--out-yaml",
            str(out_yaml),
            "--qc-png",
            str(qc_png),
        ]
    )
    assert rc_c == 0 and out_yaml.exists() and qc_png.exists()

