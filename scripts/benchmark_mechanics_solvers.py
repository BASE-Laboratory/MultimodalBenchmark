#!/usr/bin/env python3
"""Horizon 3 3D Elasticity Solver Benchmark Suite (Willot 2015 Rotated-FFT vs Analytical Bounds & SMS).

Benchmarks:
  1. Homogeneous isotropic solid (exact E_0 & M_0 in 1 PCG iteration).
  2. Two-phase 3D Backus (1962) laminate (finite contrast & infinite contrast E_pore = 0 Voigt/Reuss).
  3. Periodic 3D spherical void (E_pore = 0) against exact Voigt/Reuss and Hashin–Shtrikman bounds.
  4. Bicontinuous porous zeolite phantom with disconnected floating solid islands across
     ersatz contrast ratios {1e-2, 1e-4, 1e-6} vs exact E_pore = 0.0 (and side-by-side SMS when compiled).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from multimodal_benchmark.metrics.fft_elasticity import (
    hashin_shtrikman_isotropic_bounds,
    homogenize_elastic_fft,
    lame_parameters,
    laminate_exact_stiffness,
    uniaxial_moduli_from_normal_stiffness,
    voigt_reuss_bounds,
)
from multimodal_benchmark.metrics.mechanics import is_sms_available


def build_porous_zeolite_phantom(
    shape_zyx: tuple[int, int, int] = (32, 32, 32),
    target_porosity: float = 0.35,
    add_floating_island: bool = True,
    seed: int = 2026,
) -> np.ndarray:
    """Synthesize a reproducible 3D bicontinuous porous solid with an optional disconnected solid island."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal(shape_zyx), sigma=2.2, mode="wrap")
    thresh = float(np.quantile(field, target_porosity))
    solid = (field >= thresh).astype(np.uint8)
    if add_floating_island:
        cz, cy, cx = shape_zyx[0] // 2, shape_zyx[1] // 2, shape_zyx[2] // 2
        solid[cz - 3 : cz + 4, cy - 3 : cy + 4, cx - 3 : cx + 4] = 0
        solid[cz - 1 : cz + 2, cy - 1 : cy + 2, cx - 1 : cx + 2] = 1
    return solid


def run_benchmark_suite(grid_n: int = 32) -> dict[str, Any]:
    """Execute analytical + porous-media elasticity benchmarks and return structured JSON results."""
    e0 = 10.0e9
    nu0 = 0.25
    lam0, mu0 = lame_parameters(e0, nu0)
    m0_exact = lam0 + 2.0 * mu0
    results: dict[str, Any] = {
        "grid_n": int(grid_n),
        "youngs_modulus_pa": e0,
        "poisson_ratio": nu0,
        "sms_compiled_available": bool(is_sms_available()),
        "cases": {},
    }

    # 1. Homogeneous solid
    hom_vol = np.ones((grid_n, grid_n, grid_n), dtype=np.uint8)
    r_hom = homogenize_elastic_fft(hom_vol, youngs_modulus_pa=e0, poisson_ratio=nu0, rtol=1e-10)
    results["cases"]["homogeneous"] = {
        "e_eff_z_pa": r_hom.e_eff_xyz_pa[2],
        "e_exact_pa": e0,
        "rel_err_e": abs(r_hom.e_eff_xyz_pa[2] - e0) / e0,
        "m_confined_z_pa": r_hom.m_confined_xyz_pa[2],
        "m_exact_pa": m0_exact,
        "rel_err_m": abs(r_hom.m_confined_xyz_pa[2] - m0_exact) / m0_exact,
        "iterations_per_load": list(r_hom.iterations_per_load),
        "wall_time_s": r_hom.wall_time_s,
    }

    # 2. Bilayer Backus laminate (contrast 0.1 and exact void 0.0)
    lam_vol = np.zeros((grid_n, grid_n, grid_n), dtype=np.uint8)
    lam_vol[: grid_n // 2, :, :] = 1
    c_exact_10x = laminate_exact_stiffness(0.5, e0, nu0, 0.1 * e0, nu0, normal_axis="z")
    e_exact_10x = uniaxial_moduli_from_normal_stiffness(c_exact_10x[:3, :3])
    r_lam_10x = homogenize_elastic_fft(
        lam_vol,
        youngs_modulus_pa=e0,
        poisson_ratio=nu0,
        void_contrast=0.1,
        full_6x6=True,
        rtol=1e-10,
    )
    assert r_lam_10x.c_hom_6x6_pa is not None
    c_rel_err = float(
        np.linalg.norm(r_lam_10x.c_hom_6x6_pa - c_exact_10x) / np.linalg.norm(c_exact_10x)
    )

    r_lam_void = homogenize_elastic_fft(
        lam_vol,
        youngs_modulus_pa=e0,
        poisson_ratio=nu0,
        void_contrast=0.0,
        full_6x6=False,
        rtol=1e-10,
    )
    results["cases"]["laminate_z"] = {
        "contrast_10x_c6x6_rel_err": c_rel_err,
        "contrast_10x_e_xyz_pa": list(r_lam_10x.e_eff_xyz_pa),
        "contrast_10x_e_exact_xyz_pa": list(e_exact_10x),
        "contrast_10x_iters": list(r_lam_10x.iterations_per_load),
        "infinite_contrast_voigt_e_x_pa": r_lam_void.e_eff_xyz_pa[0],
        "infinite_contrast_voigt_exact_pa": 0.5 * e0,
        "infinite_contrast_reuss_e_z_pa": r_lam_void.e_eff_xyz_pa[2],
        "infinite_contrast_reuss_exact_pa": 0.0,
        "infinite_contrast_iters": list(r_lam_void.iterations_per_load),
    }

    # 3. Periodic spherical void (porosity ~ 0.15) vs Hashin-Shtrikman & Voigt/Reuss bounds
    zz, yy, xx = np.meshgrid(
        np.arange(grid_n) - (grid_n - 1) / 2.0,
        np.arange(grid_n) - (grid_n - 1) / 2.0,
        np.arange(grid_n) - (grid_n - 1) / 2.0,
        indexing="ij",
    )
    rad = 0.33 * grid_n
    sph_void = ((zz**2 + yy**2 + xx**2) >= rad**2).astype(np.uint8)
    r_sph = homogenize_elastic_fft(
        sph_void, youngs_modulus_pa=e0, poisson_ratio=nu0, void_contrast=0.0, full_6x6=True, rtol=1e-6
    )
    assert r_sph.c_hom_6x6_pa is not None
    hs_sph = hashin_shtrikman_isotropic_bounds(r_sph.solid_fraction, e0, nu0, 0.0, 0.0)
    vr_sph = voigt_reuss_bounds(r_sph.solid_fraction, e0, nu0, 0.0, 0.0)
    k_sph = float(np.sum(r_sph.c_norm_3x3_pa) / 9.0)
    c11 = float(np.mean(np.diag(r_sph.c_norm_3x3_pa)))
    c12 = float((np.sum(r_sph.c_norm_3x3_pa) - 3.0 * c11) / 6.0)
    c44 = float(np.mean(np.diag(r_sph.c_hom_6x6_pa[3:, 3:])))
    g_vrh = 0.5 * (5.0 / (4.0 / (c11 - c12) + 3.0 / c44) + ((c11 - c12) + 3.0 * c44) / 5.0)
    e_vrh = (9.0 * k_sph * g_vrh) / (3.0 * k_sph + g_vrh)
    e_sph_100 = float(np.mean(r_sph.e_eff_xyz_pa))
    results["cases"]["spherical_void"] = {
        "solid_fraction": r_sph.solid_fraction,
        "k_eff_cubic_pa": k_sph,
        "k_hs_upper_pa": hs_sph["k_hs_upper_pa"],
        "e_vrh_isotropic_avg_pa": e_vrh,
        "e_100_axial_pa": e_sph_100,
        "e_reuss_pa": vr_sph["e_reuss_pa"],
        "e_hs_lower_pa": hs_sph["e_hs_lower_pa"],
        "e_hs_upper_pa": hs_sph["e_hs_upper_pa"],
        "e_voigt_pa": vr_sph["e_voigt_pa"],
        "k_and_e_vrh_strictly_within_hs_bounds": bool(
            hs_sph["k_hs_lower_pa"] <= k_sph <= hs_sph["k_hs_upper_pa"]
            and hs_sph["e_hs_lower_pa"] <= e_vrh <= hs_sph["e_hs_upper_pa"]
        ),
        "iterations_per_load": list(r_sph.iterations_per_load),
        "wall_time_s": r_sph.wall_time_s,
    }

    # 4. Porous zeolite phantom with floating solid island across void_contrast sweep
    zeo = build_porous_zeolite_phantom((grid_n, grid_n, grid_n), target_porosity=0.35, add_floating_island=True)
    hs_zeo = hashin_shtrikman_isotropic_bounds(float(np.mean(zeo)), e0, nu0, 0.0, 0.0)
    sweep_rows: list[dict[str, Any]] = []
    for vc in (1e-2, 1e-4, 1e-6, 0.0):
        res_vc = homogenize_elastic_fft(
            zeo,
            youngs_modulus_pa=e0,
            poisson_ratio=nu0,
            void_contrast=vc,
            rtol=1e-6,
            max_iter=500,
        )
        row: dict[str, Any] = {
            "solver": "fft_willot2015",
            "void_contrast": vc,
            "e_eff_xyz_pa": list(res_vc.e_eff_xyz_pa),
            "e_eff_mean_pa": float(np.mean(res_vc.e_eff_xyz_pa)),
            "e_eff_norm_mean": float(np.mean(res_vc.e_eff_xyz_pa) / e0),
            "iterations_per_load": list(res_vc.iterations_per_load),
            "total_cg_iterations": int(sum(res_vc.iterations_per_load)),
            "max_relative_residual": res_vc.max_relative_residual,
            "wall_time_s": res_vc.wall_time_s,
            "converged": res_vc.converged,
        }
        sweep_rows.append(row)

    e_exact_zero = sweep_rows[-1]["e_eff_mean_pa"]
    for row in sweep_rows:
        row["rel_diff_vs_zero_contrast"] = abs(row["e_eff_mean_pa"] - e_exact_zero) / max(e_exact_zero, 1e-30)

    sms_rows: list[dict[str, Any]] = []
    if is_sms_available():
        import sms  # type: ignore[import-not-found]

        for vc in (1e-2, 1e-4, 1e-6):
            t_sms = time.perf_counter()
            e_sms_axes = []
            iters_sms = []
            conv_sms = True
            for ax in ("x", "y", "z"):
                ev, sres = sms.effective_modulus(
                    zeo,
                    dx=1e-6,
                    direction=ax,
                    youngs_modulus=e0,
                    poisson_ratio=nu0,
                    void_contrast=vc,
                    rtol=1e-5,
                    max_iter=4000,
                    check_converged=False,
                )
                e_sms_axes.append(float(ev))
                iters_sms.append(int(sres.iterations))
                conv_sms = conv_sms and bool(sres.converged)
            dt_sms = time.perf_counter() - t_sms
            sms_rows.append(
                {
                    "solver": "sms_fvm_sor",
                    "void_contrast": vc,
                    "e_eff_xyz_pa": e_sms_axes,
                    "e_eff_mean_pa": float(np.mean(e_sms_axes)),
                    "iterations_per_load": iters_sms,
                    "wall_time_s": dt_sms,
                    "converged": conv_sms,
                }
            )

    results["cases"]["porous_zeolite_with_floating_island"] = {
        "solid_fraction": float(np.mean(zeo)),
        "e_hs_upper_pa": hs_zeo["e_hs_upper_pa"],
        "fft_contrast_sweep": sweep_rows,
        "sms_contrast_sweep": sms_rows,
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Willot (2015) rotated-FFT elasticity vs analytical bounds & SMS.")
    parser.add_argument("--grid-n", type=int, default=32, help="Cubic grid resolution N (N^3 voxels)")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    report = run_benchmark_suite(grid_n=args.grid_n)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
