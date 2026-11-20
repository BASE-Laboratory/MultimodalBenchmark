"""Unit tests verifying conservative ("do-no-harm") ring, cupping, ruby-shadow, and Fresnel-fringe cleanup."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, laplace

from preprocessing.artifact_cleanup import (
    compute_ruby_beam_hardening_clean_z_mask,
    detect_ring_center_yx,
    detrend_local_tomo_cupping_3d,
    interpolate_missing_xrdct_projection,
    remove_rings_polar_conservative,
    segment_pores_fringe_guarded,
)


def test_conservative_polar_deringing_removes_full_and_partial_arcs_without_blurring_pores() -> None:
    """Verify full rings and 120-deg partial arcs are suppressed while sharp walls & pores stay intact."""
    rng = np.random.default_rng(7)
    ny, nx = 128, 128
    yy, xx = np.indices((ny, nx), dtype=np.float32)

    # Clean base microstructure: realistic non-circular catalyst fragment + sharp small pores (3-5 px)
    clean = np.full((ny, nx), 60.0, dtype=np.float32)
    pellet = (((yy - 62.0) / 46.0) ** 2 + ((xx - 66.0) / 33.0) ** 2) <= 1.0
    clean[pellet] = 85.0
    for py, px, pr in [(50, 58, 3.2), (68, 74, 2.8), (58, 82, 3.5), (75, 55, 3.0), (45, 72, 2.6)]:
        clean[((yy - py) ** 2 + (xx - px) ** 2) <= pr**2] = 61.0
    clean = gaussian_filter(clean, sigma=0.7) + rng.normal(0.0, 0.4, size=(ny, nx)).astype(np.float32)

    # Inject concentric full rings + a 120-deg partial drift arc centered at (68.0, 52.0)
    cy_true, cx_true = 68.0, 52.0
    rr = np.hypot(yy - cy_true, xx - cx_true)
    th = np.arctan2(yy - cy_true, xx - cx_true)
    full_rings = 2.4 * np.exp(-0.5 * ((rr - 24.0) / 1.2) ** 2) - 2.1 * np.exp(-0.5 * ((rr - 39.0) / 1.4) ** 2)
    partial_arc = 2.6 * np.exp(-0.5 * ((rr - 31.5) / 1.3) ** 2) * ((th > -0.8) & (th < 1.3)).astype(np.float32)
    corrupted = clean + full_rings + partial_arc

    det_cy, det_cx, _ = detect_ring_center_yx(corrupted, search_y=(20, 110), search_x=(20, 110), radial_sigma_px=4.5)
    assert np.hypot(det_cy - cy_true, det_cx - cx_true) <= 2.5

    restored, rep = remove_rings_polar_conservative(
        corrupted,
        center_yx=(cy_true, cx_true),
        radial_sigma_px=4.5,
        angular_window_deg=45.0,
    )

    err_before = float(np.sqrt(np.mean((corrupted - clean) ** 2)))
    err_after = float(np.sqrt(np.mean((restored - clean) ** 2)))
    assert err_after < 0.65 * err_before
    assert 0.98 <= rep.edge_sharpness_retention_ratio <= 1.02
    assert rep.ring_variance_reduction_pct > 40.0

    # Measure TRUE edge sharpness retention directly against the ring-free ground-truth phantom (`clean`)
    g_clean = np.hypot(
        gaussian_filter(clean, sigma=1.0, order=[1, 0]),
        gaussian_filter(clean, sigma=1.0, order=[0, 1]),
    )
    g_rest = np.hypot(
        gaussian_filter(restored, sigma=1.0, order=[1, 0]),
        gaussian_filter(restored, sigma=1.0, order=[0, 1]),
    )
    gt_edges = g_clean >= np.percentile(g_clean, 90.0)
    true_edge_ratio = float(np.mean(g_rest[gt_edges]) / np.mean(g_clean[gt_edges]))
    assert 0.98 <= true_edge_ratio <= 1.02

    # Verify polar de-ringing preserves a smooth radial Zn/Na core-shell attenuation gradient (>92% slope retained)
    r_ell = np.sqrt(((yy - 62.0) / 46.0) ** 2 + ((xx - 66.0) / 33.0) ** 2)
    cs_chem = 6.0 * np.clip(r_ell, 0.0, 1.0) ** 2 * pellet.astype(np.float32)
    rest_cs, _ = remove_rings_polar_conservative(
        corrupted + cs_chem, center_yx=(cy_true, cx_true), radial_sigma_px=4.5, angular_window_deg=45.0
    )
    recovered_cs = (rest_cs - restored)[pellet]
    slope_cs = float(np.polyfit(cs_chem[pellet], recovered_cs, 1)[0])
    assert slope_cs > 0.92


def test_off_axis_ring_removal_cupping_detrend_and_fringe_segmentation() -> None:
    """Verify off-axis crop ring removal (cx < 0), radial cupping detrending, ruby z-mask, and fringe binarization."""
    ny, nx = 96, 96
    yy, xx = np.indices((ny, nx), dtype=np.float32)

    # Off-axis ring center outside crop at (48.0, -30.0), matching 169065 local-tomography geometry
    rr_off = np.hypot(yy - 48.0, xx - (-30.0))
    base = np.full((8, ny, nx), 75.0, dtype=np.float32)
    arc = 2.2 * np.sin(rr_off * 0.9) * np.exp(-0.5 * ((rr_off - 78.0) / 14.0) ** 2)
    cupping = 9.0 * (((yy - 48.0) ** 2 + (xx - 48.0) ** 2) / (48.0**2))
    vol = base + arc[None, :, :] + cupping[None, :, :]

    derung, rep = remove_rings_polar_conservative(vol, center_yx=(48.0, -30.0), radial_sigma_px=5.0)
    assert rep.ring_rms_removed > 0.25

    leveled = detrend_local_tomo_cupping_3d(derung, sigma_yx_vox=28.0)
    assert float(np.std(leveled)) < 0.45 * float(np.std(vol))

    # Ruby beam-hardening z-mask check
    z_mask = compute_ruby_beam_hardening_clean_z_mask(500, sphere_center_z_vox=90.5, sphere_radius_vox=96.4, safety_margin_vox=12)
    assert not bool(z_mask[90])
    assert not bool(z_mask[195])
    assert bool(z_mask[210])

    # Fresnel phase-fringe binarization stability check
    zz, y3, x3 = np.indices((32, 32, 32), dtype=np.float32)
    true_solid = (((zz - 16) ** 2 + (y3 - 16) ** 2 + (x3 - 16) ** 2) > 7.5**2).astype(np.float32)
    clean_3d = 60.0 + 20.0 * gaussian_filter(true_solid, sigma=0.8)
    # Add synthetic unretrieved Fresnel bright/dark Laplacian edge overshoot!
    fringe_3d = clean_3d - 1.8 * laplace(clean_3d)
    pores_reg, _ = segment_pores_fringe_guarded(fringe_3d, tie_homogeneity_alpha_vox2=0.8)
    true_pores = (true_solid < 0.5).astype(np.uint8)
    dice = 2.0 * float(np.sum(pores_reg & true_pores)) / float(np.sum(pores_reg) + np.sum(true_pores) + 1e-8)
    assert dice > 0.92

    # Missing 80th XRD-CT projection gap interpolation & 0/360 duplicate closure merge
    ang79_gap = np.linspace(0.0, 350.886, 79)
    sino79 = np.ones((4, 79, 40), dtype=np.float32) * 3.0
    sino80, ang80 = interpolate_missing_xrdct_projection(sino79, ang79_gap, expected_step_deg=4.55696)
    assert sino80.shape == (4, 80, 40)
    assert ang80.shape == (80,)
    assert np.allclose(sino80[:, -1, :], 3.0)

    ang80_dup = np.linspace(0.0, 360.0, 80)
    sino80_dup = np.ones((4, 80, 40), dtype=np.float32) * 2.0
    sino80_dup[:, -1, :] = 4.0
    sino79_merged, ang79_merged = interpolate_missing_xrdct_projection(sino80_dup, ang80_dup, expected_step_deg=4.55696)
    assert sino79_merged.shape == (4, 79, 40)
    assert ang79_merged.shape == (79,)
    assert np.allclose(sino79_merged[:, 0, :], 3.0)


