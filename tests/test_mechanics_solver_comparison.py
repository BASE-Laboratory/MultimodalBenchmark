"""Analytical verification and regression tests for 3D elasticity homogenisation & SMS availability."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from basecoat import Volume, VolumeSet, write_ome_zarr

from multimodal_benchmark.evaluator import evaluate_bundles, load_bundle
from multimodal_benchmark.metrics import (
    compute_mechanics_discrepancy,
    effective_modulus_fft,
    hashin_shtrikman_isotropic_bounds,
    homogenize_elastic_fft,
    is_sms_available,
    laminate_exact_stiffness,
    voigt_reuss_bounds,
)
from multimodal_benchmark.metrics.fft_elasticity import (
    _rotated_wavevectors_rfft,
    lame_parameters,
    uniaxial_moduli_from_normal_stiffness,
)


def test_is_sms_available_calls_load_library_not_load_lib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: ``is_sms_available()`` must call ``sms._core._load_library()``."""
    mock_sms = ModuleType("sms")
    mock_core = ModuleType("sms._core")
    called = {"count": 0}

    def _fake_load_library() -> object:
        called["count"] += 1
        return object()

    mock_core._load_library = _fake_load_library  # type: ignore[attr-defined]
    mock_sms._core = mock_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sms", mock_sms)
    monkeypatch.setitem(sys.modules, "sms._core", mock_core)

    assert is_sms_available() is True
    assert called["count"] == 1


@pytest.mark.parametrize("shape_zyx", [(8, 10, 12), (9, 11, 13)])
def test_rotated_fft_operator_exact_self_adjointness(shape_zyx: tuple[int, int, int]) -> None:
    """Verify ``<u, A v> == <A u, v>`` to IEEE-754 float64 precision on both even and odd 3D grids."""
    rng = np.random.default_rng(42)
    dx, dy, dz = _rotated_wavevectors_rfft(shape_zyx)
    d_vec = (dx, dy, dz)
    d_conj = (np.conj(dx), np.conj(dy), np.conj(dz))

    # Inhomogeneous stiffness with exact zero-stiffness void region
    mask = (rng.random(shape_zyx) > 0.35).astype(np.float64)
    lam_x = 0.8 * mask
    two_mu_x = 1.1 * mask

    def apply_a(u: np.ndarray) -> np.ndarray:
        uh = [np.fft.rfftn(u[i], s=shape_zyx, axes=(0, 1, 2)) for i in range(3)]
        exx = np.fft.irfftn(d_vec[0] * uh[0], s=shape_zyx, axes=(0, 1, 2))
        eyy = np.fft.irfftn(d_vec[1] * uh[1], s=shape_zyx, axes=(0, 1, 2))
        ezz = np.fft.irfftn(d_vec[2] * uh[2], s=shape_zyx, axes=(0, 1, 2))
        gyz = np.fft.irfftn(d_vec[2] * uh[1] + d_vec[1] * uh[2], s=shape_zyx, axes=(0, 1, 2))
        gxz = np.fft.irfftn(d_vec[2] * uh[0] + d_vec[0] * uh[2], s=shape_zyx, axes=(0, 1, 2))
        gxy = np.fft.irfftn(d_vec[1] * uh[0] + d_vec[0] * uh[1], s=shape_zyx, axes=(0, 1, 2))
        tr = exx + eyy + ezz
        sxx = lam_x * tr + two_mu_x * exx
        syy = lam_x * tr + two_mu_x * eyy
        szz = lam_x * tr + two_mu_x * ezz
        syz = 0.5 * two_mu_x * gyz
        sxz = 0.5 * two_mu_x * gxz
        sxy = 0.5 * two_mu_x * gxy
        sh = [np.fft.rfftn(s, s=shape_zyx, axes=(0, 1, 2)) for s in (sxx, syy, szz, syz, sxz, sxy)]
        out = np.empty_like(u)
        out[0] = np.fft.irfftn(d_conj[0] * sh[0] + d_conj[1] * sh[5] + d_conj[2] * sh[4], s=shape_zyx, axes=(0, 1, 2))
        out[1] = np.fft.irfftn(d_conj[0] * sh[5] + d_conj[1] * sh[1] + d_conj[2] * sh[3], s=shape_zyx, axes=(0, 1, 2))
        out[2] = np.fft.irfftn(d_conj[0] * sh[4] + d_conj[1] * sh[3] + d_conj[2] * sh[2], s=shape_zyx, axes=(0, 1, 2))
        return out

    u = rng.standard_normal((3, *shape_zyx))
    v = rng.standard_normal((3, *shape_zyx))
    au = apply_a(u)
    av = apply_a(v)
    lhs = float(np.sum(u * av))
    rhs = float(np.sum(au * v))
    assert lhs == pytest.approx(rhs, rel=1e-13, abs=1e-13)
    assert float(np.sum(u * au)) >= -1e-13


def test_homogeneous_and_backus_laminates_exact() -> None:
    """Verify machine-precision agreement on homogeneous media and 3D Backus/Voigt/Reuss laminates."""
    e0 = 25.0e9
    nu0 = 0.30
    lam0, mu0 = lame_parameters(e0, nu0)

    # 1. Homogeneous cube
    hom = np.ones((12, 12, 12), dtype=np.uint8)
    e_unconf, r_hom = effective_modulus_fft(hom, direction="z", confined=False, youngs_modulus=e0, poisson_ratio=nu0, rtol=1e-12)
    m_conf, _ = effective_modulus_fft(hom, direction="z", confined=True, youngs_modulus=e0, poisson_ratio=nu0, rtol=1e-12)
    assert e_unconf == pytest.approx(e0, rel=1e-11)
    assert m_conf == pytest.approx(lam0 + 2.0 * mu0, rel=1e-11)
    assert max(r_hom.iterations_per_load) <= 1

    # 2. Finite-contrast 6x6 Backus laminate across all 3 normal axes
    for axis_idx, axis_name in enumerate(("z", "y", "x")):
        lam_vol = np.zeros((16, 16, 16), dtype=np.uint8)
        slc = [slice(None), slice(None), slice(None)]
        slc[axis_idx] = slice(0, 6)  # phi_1 = 6/16 = 0.375
        lam_vol[tuple(slc)] = 1

        c_exact = laminate_exact_stiffness(0.375, e0, nu0, 0.05 * e0, nu0, normal_axis=axis_name)
        e_exact = uniaxial_moduli_from_normal_stiffness(c_exact[:3, :3])
        res = homogenize_elastic_fft(
            lam_vol,
            youngs_modulus_pa=e0,
            poisson_ratio=nu0,
            void_contrast=0.05,
            full_6x6=True,
            rtol=1e-11,
        )
        assert res.c_hom_6x6_pa is not None
        assert np.allclose(res.c_hom_6x6_pa, c_exact, rtol=1e-9, atol=1e-2)
        assert res.e_eff_xyz_pa == pytest.approx(e_exact, rel=1e-9)

    # 3. Infinite-contrast (E_pore = 0) parallel/series laminate
    void_lam = np.zeros((12, 12, 12), dtype=np.uint8)
    void_lam[:6, :, :] = 1
    res_void = homogenize_elastic_fft(void_lam, youngs_modulus_pa=e0, poisson_ratio=nu0, void_contrast=0.0, rtol=1e-11)
    assert res_void.e_eff_xyz_pa[0] == pytest.approx(0.5 * e0, rel=1e-10)
    assert res_void.e_eff_xyz_pa[1] == pytest.approx(0.5 * e0, rel=1e-10)
    assert res_void.e_eff_xyz_pa[2] == pytest.approx(0.0, abs=1e-4)


def test_spherical_void_and_porous_floating_island_bounds_and_evaluator(tmp_path: Path) -> None:
    """Verify cubic bulk Hashin-Shtrikman bounds, Voigt/Reuss bounds, floating-island stability, and evaluator."""
    e0 = 10.0e9
    nu0 = 0.25
    n = 24
    zz, yy, xx = np.meshgrid(
        np.arange(n) - (n - 1) / 2.0,
        np.arange(n) - (n - 1) / 2.0,
        np.arange(n) - (n - 1) / 2.0,
        indexing="ij",
    )
    sph_void = ((zz**2 + yy**2 + xx**2) >= (0.32 * n) ** 2).astype(np.uint8)
    res_sph = homogenize_elastic_fft(sph_void, youngs_modulus_pa=e0, poisson_ratio=nu0, void_contrast=0.0, full_6x6=True, rtol=1e-6)
    assert res_sph.c_hom_6x6_pa is not None
    hs = hashin_shtrikman_isotropic_bounds(res_sph.solid_fraction, e0, nu0, 0.0, 0.0)
    vr = voigt_reuss_bounds(res_sph.solid_fraction, e0, nu0, 0.0, 0.0)

    # For a cubic crystal, bulk modulus K_eff = sum(C_norm)/9 and Voigt-averaged shear G_voigt_cubic = ((C11-C12) + 3*C44)/5
    k_eff = float(np.sum(res_sph.c_norm_3x3_pa) / 9.0)
    c11 = float(np.mean(np.diag(res_sph.c_norm_3x3_pa)))
    c12 = float((np.sum(res_sph.c_norm_3x3_pa) - 3.0 * c11) / 6.0)
    c44 = float(np.mean(np.diag(res_sph.c_hom_6x6_pa[3:, 3:])))
    g_reuss_orient = 5.0 / (4.0 / (c11 - c12) + 3.0 / c44)
    g_voigt_orient = ((c11 - c12) + 3.0 * c44) / 5.0
    g_vrh = 0.5 * (g_reuss_orient + g_voigt_orient)
    e_vrh = (9.0 * k_eff * g_vrh) / (3.0 * k_eff + g_vrh)

    assert hs["k_hs_lower_pa"] <= k_eff <= hs["k_hs_upper_pa"] <= vr["k_voigt_pa"]
    assert hs["e_hs_lower_pa"] <= e_vrh <= hs["e_hs_upper_pa"] <= vr["e_voigt_pa"]
    e_100 = float(np.mean(res_sph.e_eff_xyz_pa))
    assert vr["e_reuss_pa"] < e_100 < vr["e_voigt_pa"]
    assert res_sph.converged is True

    # Floating solid island inside pore cavity with exact E_pore = 0
    porous = sph_void.copy()
    porous[10:14, 10:14, 10:14] = 1
    res_island = homogenize_elastic_fft(porous, youngs_modulus_pa=e0, poisson_ratio=nu0, void_contrast=0.0, rtol=1e-6)
    assert res_island.converged is True
    assert np.all(np.isfinite(res_island.e_eff_xyz_pa))

    # Direct discrepancy & end-to-end OME-Zarr evaluator using built-in FFT backend
    disc = compute_mechanics_discrepancy(
        porous,
        sph_void,
        inputs_are_grayscale=False,
        voxel_size_um=0.8,
        backend="fft",
    )
    assert disc["sms_converged_all_axes_pred"] == 1.0
    assert disc["sms_converged_all_axes_ref"] == 1.0
    assert disc["e_eff_mean_ref_pa"] > 0.0

    ref_store = tmp_path / "ref_t4a.ome.zarr"
    pred_store = tmp_path / "pred_t4a.ome.zarr"
    write_ome_zarr(VolumeSet(image=Volume.from_array(sph_void, voxel_size_um=0.54)), ref_store)
    write_ome_zarr(VolumeSet(image=Volume.from_array(porous, voxel_size_um=0.54)), pred_store)
    ev = evaluate_bundles(
        "t4a",
        load_bundle(pred_store),
        load_bundle(ref_store),
        with_transport=False,
        with_mechanics=True,
    )
    assert ev["metrics"]["e_eff_mean_ref_pa"] > 0.0
    assert np.isfinite(ev["metrics"]["e_eff_rel_err"])
