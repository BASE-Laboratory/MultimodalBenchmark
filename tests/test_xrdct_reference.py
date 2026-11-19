"""Unit and real-data tests for Stage 0A XRD-CT COR estimation, NNLS unmixing, FRC noise floor, and manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage.transform import radon

from multimodal_benchmark.io import load_ascii_xrd, load_xrdct_peak_maps, load_xrdct_volume_50um
from multimodal_benchmark.manifest import load_manifest, resolve_doi
from registration.xrdct_reference import (
    audit_cached_xrdct_outputs,
    estimate_conjugate_cor,
    estimate_odd_even_noise_floor,
    legacy_two_bin_window_sinogram,
    unmix_sinogram_nnls,
)


def test_manifest_resolves_broken_diad_doi() -> None:
    assert resolve_doi("10.5281/zenodo.13327988") == "10.5281/zenodo.13349402"
    manifest = load_manifest()
    rec = manifest.get_record("10.5281/zenodo.13327988")
    assert rec["record_id"] == "13349402"
    assert rec["hdf5_dataset"] == "4-AstraReconGpu-tomo/data"


def test_conjugate_cor_79_projections_subpixel_recovery() -> None:
    """Verify sub-pixel COR recovery on a 79-projection 0-360 scan (matching k11-43322)."""
    n = 40
    yy, xx = np.ogrid[:n, :n]
    phantom = (((yy - 18.0) ** 2 + (xx - 22.0) ** 2) <= 9.0 ** 2).astype(np.float64)
    phantom += 1.5 * (((yy - 14.0) ** 2 + (xx - 16.0) ** 2) <= 3.5 ** 2)

    # 80 angles over [0, 360), dropping projection #37 to mimic 79-angle scan k11-43322
    angles_80 = np.linspace(0.0, 360.0, 80, endpoint=False)
    angles_79 = np.delete(angles_80, 37)

    # radon() outputs (n_trans, n_angles) centered at nominal (n-1)/2 = 19.5
    sino_tn = radon(phantom, theta=angles_79, circle=True)
    sino_at = sino_tn.T  # (79, 40)

    true_delta_px = 1.65
    from scipy.ndimage import shift as nd_shift

    shifted_sino = nd_shift(sino_at, shift=(0.0, true_delta_px), order=3, mode="nearest")
    cor_res = estimate_conjugate_cor(shifted_sino, angles_79)

    assert cor_res.num_conjugate_pairs >= 38
    assert abs(cor_res.offset_from_nominal_px - true_delta_px) < 0.05
    assert abs(cor_res.center_px - (20.0 + true_delta_px)) < 0.05


def test_nnls_unmixing_beats_legacy_two_bin_window_on_close_doublet() -> None:
    """Verify multi-peak NNLS unmixing separates 0.005 A^-1 doublet + capillary halo far cleaner than 2-bin window."""
    rng = np.random.default_rng(7)
    q = np.linspace(0.9, 3.0, 600)

    def _gauss(mu: float, sig: float) -> np.ndarray:
        return np.exp(-0.5 * ((q - mu) / sig) ** 2)

    # Realistic synchrotron doublet overlap (sigma=0.007 A^-1 across a 0.0049 A^-1 peak shift) + secondary peaks
    na_basis = _gauss(1.6506, 0.007) + 0.7 * _gauss(1.097, 0.007) + 0.6 * _gauss(2.360, 0.007)
    zn_basis = _gauss(1.6555, 0.007) + 0.7 * _gauss(1.102, 0.007) + 0.6 * _gauss(2.625, 0.007)
    cap_basis = 0.6 * _gauss(1.58, 0.18)

    n_ang, n_tr = 12, 16
    na_true = np.clip(rng.uniform(0.0, 2.0, size=(n_ang, n_tr)), 0.0, None)
    zn_true = np.clip(rng.uniform(0.0, 2.0, size=(n_ang, n_tr)), 0.0, None)
    zn_true[na_true > 1.0] = 0.0
    cap_true = rng.uniform(0.5, 2.0, size=(n_ang, n_tr))

    sino_qat = (
        na_basis[:, None, None] * na_true[None, :, :]
        + zn_basis[:, None, None] * zn_true[None, :, :]
        + cap_basis[:, None, None] * cap_true[None, :, :]
        + rng.normal(0.0, 0.015, size=(q.size, n_ang, n_tr))
    )

    nnls_res = unmix_sinogram_nnls(sino_qat, q, na_basis, zn_basis, cap_basis)
    leg_res = legacy_two_bin_window_sinogram(sino_qat, q, na_q=1.6506, zn_q=1.6555)

    r_nnls_na = float(np.corrcoef(nnls_res["na"].ravel(), na_true.ravel())[0, 1])
    r_leg_na = float(np.corrcoef(leg_res["na"].ravel(), na_true.ravel())[0, 1])
    assert r_nnls_na > 0.99
    assert r_nnls_na > r_leg_na + 0.15


def test_odd_even_frc_noise_floor() -> None:
    n = 40
    yy, xx = np.ogrid[:n, :n]
    img = (((yy - 20.0) ** 2 + (xx - 20.0) ** 2) <= 12.0 ** 2).astype(np.float64)
    angles = np.linspace(0.0, 360.0, 80, endpoint=False)
    rng = np.random.default_rng(123)
    sino = radon(img, theta=angles, circle=True).T + rng.normal(0.0, 0.25, size=(80, n))

    rec, sigma2, frc = estimate_odd_even_noise_floor(sino, angles)
    assert rec.shape == (n, n)
    assert sigma2.shape == (n, n)
    assert np.all(sigma2 > 0)
    assert 0.05 < frc.cutoff_0p5_cyc_per_px <= 0.5


@pytest.mark.skipif(
    not Path("data/cache/zenodo/13Xhighres_2-Theta_Omega.asc").exists(),
    reason="Cached Zenodo files not present",
)
def test_real_zenodo_powder_and_xrdct_audit(tmp_path: Path) -> None:
    spec_na = load_ascii_xrd("data/cache/zenodo/13Xhighres_2-Theta_Omega.asc")
    assert spec_na.q_inv_angstrom.size == 7501

    peaks_25 = load_xrdct_peak_maps("data/cache/zenodo/output_43324_zn.nxs")
    assert len(peaks_25) == 13
    assert next(iter(peaks_25.values())).shape == (40, 40)

    vol_50, used_ids = load_xrdct_volume_50um("data/cache/zenodo", phase="na")
    assert vol_50.shape == (20, 20, 20)
    assert len(used_ids) == 20

    rep = audit_cached_xrdct_outputs("data/cache/zenodo", out_dir=tmp_path)
    assert abs(rep["powder_xrd"]["peak_shift_inv_A"] - 0.00488) < 0.001
    assert set(rep["scans_25um"].keys()) == {"43322", "43323", "43324", "43325"}

    sino_25 = Path("data/cache/zenodo/k11-43324-diffraction-DiffInt_Mask_Azz_Txt.nxs")
    sino_50 = Path("data/cache/zenodo/k11-43345-diffraction-DiffInt_Mask_Azz_Txt_50um.nxs")
    if sino_25.exists() and sino_50.exists():
        from multimodal_benchmark.io import load_xrdct_sinogram

        s25 = load_xrdct_sinogram(sino_25, q_range=(1.648, 1.658), scan_id=43324)
        s50 = load_xrdct_sinogram(sino_50, q_range=(1.648, 1.658), scan_id=43345)
        assert s25.angles_deg[0] == pytest.approx(0.0)
        assert s50.angles_deg[0] == pytest.approx(-47.5)
        assert s25.kb_cs_x_px is not None and np.mean(s25.kb_cs_x_px) == pytest.approx(1280.5, abs=0.1)
        # Verify frame-0 physical equivalence across the -47.5 deg soft-motor offset
        p25 = s25.data.sum(0)[::2, :40].reshape(40, 20, 2).mean(-1)
        p50 = s50.data.sum(0)
        p25_d = p25 - p25.mean(axis=1, keepdims=True)
        p50_d = p50 - p50.mean(axis=1, keepdims=True)
        assert float(np.corrcoef(p25_d.ravel(), p50_d.ravel())[0, 1]) > 0.99

