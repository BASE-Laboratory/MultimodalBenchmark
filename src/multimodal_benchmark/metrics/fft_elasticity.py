"""Pure-Python Willot (2015) rotated-stencil FFT-CG 3D linear elasticity homogenisation solver.

Implements the rotated centered-difference Fourier-Galerkin discretisation of Willot,
*Comptes Rendus Mécanique* **343**(3), 232–245 (2015), doi:10.1016/j.crme.2014.12.005,
coupled with Preconditioned Conjugate Gradients (PCG) preconditioned by the exact
Sherman–Morrison inverse of the homogeneous reference acoustic tensor.

Key numerical properties:
- Exact self-adjoint positive semi-definite operator ``A = B* C(x) B`` in IEEE-754 float64
  using real-to-complex ``rfftn`` / ``irfftn`` transforms.
- Native support for infinite phase contrast (exact ``E_pore = 0`` in void voxels, and
  disconnected floating solid islands ``b ⟂ ker(A)`` without artificial ersatz stiffness).
- Exact machine-precision reproduction of 3D Backus (1962) / Voigt / Reuss layered-medium
  stiffness tensors in 1–2 PCG iterations.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from basecoat import Volume
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg


@dataclass(frozen=True)
class FFTElasticityResult:
    """Homogenized 3D elastic moduli and solver diagnostics from Willot (2015) rotated-FFT PCG."""

    c_norm_3x3_pa: NDArray[np.float64]
    c_hom_6x6_pa: NDArray[np.float64] | None
    e_eff_xyz_pa: tuple[float, float, float]
    m_confined_xyz_pa: tuple[float, float, float]
    solid_fraction: float
    converged: bool
    iterations_per_load: tuple[int, ...]
    max_relative_residual: float
    wall_time_s: float


def lame_parameters(
    youngs_modulus: float | NDArray[np.floating[Any]],
    poisson_ratio: float | NDArray[np.floating[Any]],
) -> tuple[Any, Any]:
    """Convert isotropic ``(E, nu)`` to Lamé parameters ``(lambda, mu)``."""
    lam = (youngs_modulus * poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    return lam, mu


def voigt_reuss_bounds(
    solid_fraction: float,
    youngs_modulus_solid: float,
    poisson_ratio_solid: float,
    youngs_modulus_pore: float = 0.0,
    poisson_ratio_pore: float = 0.0,
) -> dict[str, float]:
    """Exact 3D Voigt (iso-strain) and Reuss (iso-stress) bounds on bulk, shear, and Young's moduli."""
    phi1 = float(np.clip(solid_fraction, 0.0, 1.0))
    phi2 = 1.0 - phi1
    k1 = youngs_modulus_solid / (3.0 * (1.0 - 2.0 * poisson_ratio_solid))
    g1 = youngs_modulus_solid / (2.0 * (1.0 + poisson_ratio_solid))
    k2 = (
        youngs_modulus_pore / (3.0 * (1.0 - 2.0 * poisson_ratio_pore))
        if youngs_modulus_pore > 0.0
        else 0.0
    )
    g2 = (
        youngs_modulus_pore / (2.0 * (1.0 + poisson_ratio_pore))
        if youngs_modulus_pore > 0.0
        else 0.0
    )

    k_voigt = phi1 * k1 + phi2 * k2
    g_voigt = phi1 * g1 + phi2 * g2
    e_voigt = (9.0 * k_voigt * g_voigt) / max(3.0 * k_voigt + g_voigt, 1e-30)

    if phi2 > 0.0 and (k2 <= 0.0 or g2 <= 0.0):
        k_reuss = 0.0 if k2 <= 0.0 else 1.0 / (phi1 / k1 + phi2 / k2)
        g_reuss = 0.0 if g2 <= 0.0 else 1.0 / (phi1 / g1 + phi2 / g2)
        e_reuss = 0.0
    elif phi1 == 0.0:
        k_reuss, g_reuss, e_reuss = k2, g2, youngs_modulus_pore
    else:
        k_reuss = 1.0 / (phi1 / k1 + phi2 / k2)
        g_reuss = 1.0 / (phi1 / g1 + phi2 / g2)
        e_reuss = (9.0 * k_reuss * g_reuss) / max(3.0 * k_reuss + g_reuss, 1e-30)

    return {
        "k_voigt_pa": float(k_voigt),
        "g_voigt_pa": float(g_voigt),
        "e_voigt_pa": float(e_voigt),
        "k_reuss_pa": float(k_reuss),
        "g_reuss_pa": float(g_reuss),
        "e_reuss_pa": float(e_reuss),
    }


def hashin_shtrikman_isotropic_bounds(
    solid_fraction: float,
    youngs_modulus_solid: float,
    poisson_ratio_solid: float,
    youngs_modulus_pore: float = 0.0,
    poisson_ratio_pore: float = 0.0,
) -> dict[str, float]:
    """Exact 3D Hashin–Shtrikman (1963) isotropic two-phase bounds on K, G, and E."""
    phi1 = float(np.clip(solid_fraction, 0.0, 1.0))
    phi2 = 1.0 - phi1
    if phi1 <= 0.0:
        return {"k_hs_lower_pa": youngs_modulus_pore / max(3.0 * (1.0 - 2.0 * poisson_ratio_pore), 1e-30),
                "g_hs_lower_pa": youngs_modulus_pore / max(2.0 * (1.0 + poisson_ratio_pore), 1e-30),
                "e_hs_lower_pa": float(youngs_modulus_pore),
                "k_hs_upper_pa": youngs_modulus_pore / max(3.0 * (1.0 - 2.0 * poisson_ratio_pore), 1e-30),
                "g_hs_upper_pa": youngs_modulus_pore / max(2.0 * (1.0 + poisson_ratio_pore), 1e-30),
                "e_hs_upper_pa": float(youngs_modulus_pore)}
    if phi2 <= 0.0:
        k1 = youngs_modulus_solid / (3.0 * (1.0 - 2.0 * poisson_ratio_solid))
        g1 = youngs_modulus_solid / (2.0 * (1.0 + poisson_ratio_solid))
        return {"k_hs_lower_pa": float(k1), "g_hs_lower_pa": float(g1), "e_hs_lower_pa": float(youngs_modulus_solid),
                "k_hs_upper_pa": float(k1), "g_hs_upper_pa": float(g1), "e_hs_upper_pa": float(youngs_modulus_solid)}

    k1 = youngs_modulus_solid / (3.0 * (1.0 - 2.0 * poisson_ratio_solid))
    g1 = youngs_modulus_solid / (2.0 * (1.0 + poisson_ratio_solid))
    k2 = (
        youngs_modulus_pore / (3.0 * (1.0 - 2.0 * poisson_ratio_pore))
        if youngs_modulus_pore > 0.0
        else 0.0
    )
    g2 = (
        youngs_modulus_pore / (2.0 * (1.0 + poisson_ratio_pore))
        if youngs_modulus_pore > 0.0
        else 0.0
    )

    def _hs_k(k_ref: float, g_ref: float) -> float:
        if g_ref <= 0.0 and (k1 <= 0.0 or k2 <= 0.0):
            return 0.0
        k_star = (4.0 / 3.0) * g_ref
        d1 = k1 + k_star
        d2 = k2 + k_star
        if (phi1 > 0.0 and d1 <= 0.0) or (phi2 > 0.0 and d2 <= 0.0):
            return 0.0
        inv_sum = (phi1 / d1 if phi1 > 0.0 else 0.0) + (phi2 / d2 if phi2 > 0.0 else 0.0)
        return max(0.0, (1.0 / inv_sum) - k_star)

    def _hs_g(k_ref: float, g_ref: float) -> float:
        if g_ref <= 0.0 or (k_ref + 2.0 * g_ref) <= 0.0:
            return 0.0
        g_star = (g_ref * (9.0 * k_ref + 8.0 * g_ref)) / (6.0 * (k_ref + 2.0 * g_ref))
        d1 = g1 + g_star
        d2 = g2 + g_star
        if (phi1 > 0.0 and d1 <= 0.0) or (phi2 > 0.0 and d2 <= 0.0):
            return 0.0
        inv_sum = (phi1 / d1 if phi1 > 0.0 else 0.0) + (phi2 / d2 if phi2 > 0.0 else 0.0)
        return max(0.0, (1.0 / inv_sum) - g_star)

    k_lo_ref, k_hi_ref = min(k1, k2), max(k1, k2)
    g_lo_ref, g_hi_ref = min(g1, g2), max(g1, g2)
    k_lo = _hs_k(k_lo_ref, g_lo_ref)
    g_lo = _hs_g(k_lo_ref, g_lo_ref)
    k_hi = _hs_k(k_hi_ref, g_hi_ref)
    g_hi = _hs_g(k_hi_ref, g_hi_ref)

    e_lo = (9.0 * k_lo * g_lo) / max(3.0 * k_lo + g_lo, 1e-30) if (k_lo > 0 and g_lo > 0) else 0.0
    e_hi = (9.0 * k_hi * g_hi) / max(3.0 * k_hi + g_hi, 1e-30) if (k_hi > 0 and g_hi > 0) else 0.0
    return {
        "k_hs_lower_pa": float(k_lo),
        "g_hs_lower_pa": float(g_lo),
        "e_hs_lower_pa": float(e_lo),
        "k_hs_upper_pa": float(k_hi),
        "g_hs_upper_pa": float(g_hi),
        "e_hs_upper_pa": float(e_hi),
    }


def laminate_exact_stiffness(
    phi_1: float,
    e_1: float,
    nu_1: float,
    e_2: float,
    nu_2: float,
    normal_axis: str = "z",
) -> NDArray[np.float64]:
    """Exact Backus (1962) 6x6 Voigt homogenized stiffness [xx, yy, zz, yz, xz, xy] of a bilayer laminate."""
    f1 = float(phi_1)
    f2 = 1.0 - f1
    lam1, mu1 = lame_parameters(e_1, nu_1) if e_1 > 0.0 else (0.0, 0.0)
    lam2, mu2 = lame_parameters(e_2, nu_2) if e_2 > 0.0 else (0.0, 0.0)
    p1 = lam1 + 2.0 * mu1
    p2 = lam2 + 2.0 * mu2

    # Build for normal = z first, then permute axes if normal is x or y
    inv_p_avg = (0.0 if (f1 > 0 and p1 == 0) or (f2 > 0 and p2 == 0) else 1.0 / (f1 / p1 + f2 / p2))
    lam_over_p_avg = (f1 * (lam1 / p1) if p1 > 0 else 0.0) + (f2 * (lam2 / p2) if p2 > 0 else 0.0)

    c_zz = inv_p_avg
    c_xz = inv_p_avg * lam_over_p_avg
    c_xx = (
        (f1 * (p1 - lam1 * lam1 / p1) if p1 > 0 else 0.0)
        + (f2 * (p2 - lam2 * lam2 / p2) if p2 > 0 else 0.0)
        + inv_p_avg * (lam_over_p_avg**2)
    )
    c_xy = (
        (f1 * (lam1 - lam1 * lam1 / p1) if p1 > 0 else 0.0)
        + (f2 * (lam2 - lam2 * lam2 / p2) if p2 > 0 else 0.0)
        + inv_p_avg * (lam_over_p_avg**2)
    )
    c_66 = f1 * mu1 + f2 * mu2
    c_44 = 0.0 if (f1 > 0 and mu1 == 0) or (f2 > 0 and mu2 == 0) else 1.0 / (f1 / mu1 + f2 / mu2)

    c = np.zeros((6, 6), dtype=np.float64)
    n_ax = normal_axis.lower()
    if n_ax == "z":
        c[0, 0] = c[1, 1] = c_xx
        c[2, 2] = c_zz
        c[0, 1] = c[1, 0] = c_xy
        c[0, 2] = c[2, 0] = c[1, 2] = c[2, 1] = c_xz
        c[3, 3] = c[4, 4] = c_44
        c[5, 5] = c_66
    elif n_ax == "y":
        c[0, 0] = c[2, 2] = c_xx
        c[1, 1] = c_zz
        c[0, 2] = c[2, 0] = c_xy
        c[0, 1] = c[1, 0] = c[1, 2] = c[2, 1] = c_xz
        c[3, 3] = c[5, 5] = c_44
        c[4, 4] = c_66
    elif n_ax == "x":
        c[1, 1] = c[2, 2] = c_xx
        c[0, 0] = c_zz
        c[1, 2] = c[2, 1] = c_xy
        c[0, 1] = c[1, 0] = c[0, 2] = c[2, 0] = c_xz
        c[4, 4] = c[5, 5] = c_44
        c[3, 3] = c_66
    else:
        raise ValueError(f"normal_axis must be 'x', 'y', or 'z', got {normal_axis!r}")
    return c


def uniaxial_moduli_from_normal_stiffness(
    c_norm_3x3: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Compute unconfined uniaxial Young's moduli ``(E_x, E_y, E_z)`` via lateral Schur complement.

    Using ``np.linalg.pinv(..., rcond=1e-12)`` on the 2x2 transverse block avoids singularities
    on 1D-percolating pillar arrays or laminates with ``E_pore = 0``.
    """
    c = np.asarray(c_norm_3x3, dtype=np.float64)
    out: list[float] = []
    for d in range(3):
        c_dd = float(c[d, d])
        if c_dd <= 0.0:
            out.append(0.0)
            continue
        perp = [i for i in range(3) if i != d]
        c_pp = c[np.ix_(perp, perp)]
        c_dp = c[d, perp]
        c_pp_inv = np.linalg.pinv(c_pp, rcond=1e-12)
        e_d = float(c_dd - c_dp @ c_pp_inv @ c_dp)
        out.append(max(0.0, e_d))
    return (out[0], out[1], out[2])


def _rotated_wavevectors_rfft(
    shape_zyx: tuple[int, int, int],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Willot (2015) rotated-grid discrete Fourier derivative multipliers ``(d_x, d_y, d_z)`` on ``rfftn`` grid."""
    nz, ny, nx = shape_zyx
    qz = 2.0 * math.pi * np.fft.fftfreq(nz)[:, None, None]
    qy = 2.0 * math.pi * np.fft.fftfreq(ny)[None, :, None]
    qx = 2.0 * math.pi * np.fft.rfftfreq(nx)[None, None, :]

    ez_m = np.exp(1j * qz) - 1.0
    ez_p = np.exp(1j * qz) + 1.0
    ey_m = np.exp(1j * qy) - 1.0
    ey_p = np.exp(1j * qy) + 1.0
    ex_m = np.exp(1j * qx) - 1.0
    ex_p = np.exp(1j * qx) + 1.0

    dx = 0.25 * ez_p * ey_p * ex_m
    dy = 0.25 * ez_p * ey_m * ex_p
    dz = 0.25 * ez_m * ey_p * ex_p
    return dx, dy, dz


def homogenize_elastic_fft(
    solid_or_volume: NDArray[Any] | Volume,
    *,
    youngs_modulus_pa: float = 10.0e9,
    poisson_ratio: float = 0.25,
    solid_phase: int | Sequence[int] | None = 1,
    void_contrast: float = 0.0,
    full_6x6: bool = False,
    rtol: float = 1e-6,
    max_iter: int = 500,
) -> FFTElasticityResult:
    """Homogenize a 3D isotropic voxel microstructure via Willot (2015) rotated-stencil FFT-PCG.

    Parameters
    ----------
    solid_or_volume:
        3D array ``(nz, ny, nx)`` or calibrated :class:`basecoat.Volume`.
    youngs_modulus_pa:
        Solid-phase Young's modulus in Pascals (> 0).
    poisson_ratio:
        Isotropic Poisson's ratio in ``(-1.0, 0.5)``.
    solid_phase:
        Integer label(s) treated as solid. If ``None``, non-zero entries are treated as solid.
    void_contrast:
        Ratio ``E_pore / E_solid`` in ``[0, 1]``. Defaults to **exact ``0.0``** (infinite contrast).
    full_6x6:
        If ``True``, solves all 6 Voigt macroscopic strain cases ``[xx, yy, zz, yz, xz, xy]``;
        otherwise solves only the 3 axial normal strain cases ``[xx, yy, zz]`` required for
        ``(E_eff_x, E_eff_y, E_eff_z)`` and ``(M_x, M_y, M_z)``.
    rtol:
        Relative residual convergence tolerance for PCG.
    max_iter:
        Maximum PCG iterations per macroscopic strain load case.
    """
    t0 = time.perf_counter()
    if youngs_modulus_pa <= 0.0 or not math.isfinite(youngs_modulus_pa):
        raise ValueError(f"youngs_modulus_pa must be finite and positive, got {youngs_modulus_pa}")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError(f"poisson_ratio must lie strictly in (-1.0, 0.5), got {poisson_ratio}")
    if not (0.0 <= void_contrast <= 1.0):
        raise ValueError(f"void_contrast must lie in [0.0, 1.0], got {void_contrast}")

    if isinstance(solid_or_volume, Volume):
        arr = np.asarray(solid_or_volume.data)
    elif hasattr(solid_or_volume, "data") and hasattr(solid_or_volume, "spacing_in"):
        arr = np.asarray(solid_or_volume.data)
    else:
        arr = np.asarray(solid_or_volume)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume (nz, ny, nx), got shape {arr.shape}")

    if solid_phase is None:
        is_solid = arr > 0
    elif isinstance(solid_phase, int):
        is_solid = arr == solid_phase
    else:
        is_solid = np.isin(arr, list(solid_phase))

    solid_frac = float(np.mean(is_solid))
    if solid_frac <= 0.0 and void_contrast == 0.0:
        z3 = np.zeros((3, 3), dtype=np.float64)
        z6 = np.zeros((6, 6), dtype=np.float64) if full_6x6 else None
        return FFTElasticityResult(
            c_norm_3x3_pa=z3,
            c_hom_6x6_pa=z6,
            e_eff_xyz_pa=(0.0, 0.0, 0.0),
            m_confined_xyz_pa=(0.0, 0.0, 0.0),
            solid_fraction=0.0,
            converged=True,
            iterations_per_load=(0, 0, 0) if not full_6x6 else (0, 0, 0, 0, 0, 0),
            max_relative_residual=0.0,
            wall_time_s=time.perf_counter() - t0,
        )

    # Normalize stiffness by youngs_modulus_pa during PCG for optimal floating-point conditioning
    lam0, mu0 = lame_parameters(1.0, float(poisson_ratio))
    scale_field = np.where(is_solid, 1.0, float(void_contrast)).astype(np.float64)
    lam_x = lam0 * scale_field
    two_mu_x = (2.0 * mu0) * scale_field

    shape_zyx = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
    n_dof = 3 * shape_zyx[0] * shape_zyx[1] * shape_zyx[2]
    dx, dy, dz = _rotated_wavevectors_rfft(shape_zyx)
    d_vec = (dx, dy, dz)  # component order: 0=x, 1=y, 2=z
    d_conj = (np.conj(dx), np.conj(dy), np.conj(dz))

    d_norm2 = np.abs(dx) ** 2 + np.abs(dy) ** 2 + np.abs(dz) ** 2
    active_k = d_norm2 > 1e-24
    inv_d2 = np.where(active_k, 1.0 / np.where(active_k, d_norm2, 1.0), 0.0)
    # Reference medium preconditioner K_0(k)^{-1} with (lam_ref, mu_ref) = (lam0, mu0)
    alpha_sherman = (lam0 + mu0) / (lam0 + 2.0 * mu0)
    inv_mu0 = 1.0 / mu0

    def _strain_from_u(
        u_xyz: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Compute Voigt strain [exx, eyy, ezz, 2*eyz, 2*exz, 2*exy] from corner displacement u."""
        uh0 = np.fft.rfftn(u_xyz[0], s=shape_zyx, axes=(0, 1, 2))
        uh1 = np.fft.rfftn(u_xyz[1], s=shape_zyx, axes=(0, 1, 2))
        uh2 = np.fft.rfftn(u_xyz[2], s=shape_zyx, axes=(0, 1, 2))

        exx = np.fft.irfftn(d_vec[0] * uh0, s=shape_zyx, axes=(0, 1, 2))
        eyy = np.fft.irfftn(d_vec[1] * uh1, s=shape_zyx, axes=(0, 1, 2))
        ezz = np.fft.irfftn(d_vec[2] * uh2, s=shape_zyx, axes=(0, 1, 2))
        g_yz = np.fft.irfftn(d_vec[2] * uh1 + d_vec[1] * uh2, s=shape_zyx, axes=(0, 1, 2))
        g_xz = np.fft.irfftn(d_vec[2] * uh0 + d_vec[0] * uh2, s=shape_zyx, axes=(0, 1, 2))
        g_xy = np.fft.irfftn(d_vec[1] * uh0 + d_vec[0] * uh1, s=shape_zyx, axes=(0, 1, 2))
        return exx, eyy, ezz, g_yz, g_xz, g_xy

    def _constitutive_stress(
        exx: NDArray[np.float64],
        eyy: NDArray[np.float64],
        ezz: NDArray[np.float64],
        g_yz: NDArray[np.float64],
        g_xz: NDArray[np.float64],
        g_xy: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        tr = exx + eyy + ezz
        p_iso = lam_x * tr
        sxx = p_iso + two_mu_x * exx
        syy = p_iso + two_mu_x * eyy
        szz = p_iso + two_mu_x * ezz
        syz = (0.5 * two_mu_x) * g_yz
        sxz = (0.5 * two_mu_x) * g_xz
        sxy = (0.5 * two_mu_x) * g_xy
        return sxx, syy, szz, syz, sxz, sxy

    def _div_adjoint(
        sxx: NDArray[np.float64],
        syy: NDArray[np.float64],
        szz: NDArray[np.float64],
        syz: NDArray[np.float64],
        sxz: NDArray[np.float64],
        sxy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Exact L^2 adjoint ``B* sigma`` mapping cell stresses to corner nodal forces."""
        sh_xx = np.fft.rfftn(sxx, s=shape_zyx, axes=(0, 1, 2))
        sh_yy = np.fft.rfftn(syy, s=shape_zyx, axes=(0, 1, 2))
        sh_zz = np.fft.rfftn(szz, s=shape_zyx, axes=(0, 1, 2))
        sh_yz = np.fft.rfftn(syz, s=shape_zyx, axes=(0, 1, 2))
        sh_xz = np.fft.rfftn(sxz, s=shape_zyx, axes=(0, 1, 2))
        sh_xy = np.fft.rfftn(sxy, s=shape_zyx, axes=(0, 1, 2))

        out = np.empty((3, *shape_zyx), dtype=np.float64)
        out[0] = np.fft.irfftn(
            d_conj[0] * sh_xx + d_conj[1] * sh_xy + d_conj[2] * sh_xz, s=shape_zyx, axes=(0, 1, 2)
        )
        out[1] = np.fft.irfftn(
            d_conj[0] * sh_xy + d_conj[1] * sh_yy + d_conj[2] * sh_yz, s=shape_zyx, axes=(0, 1, 2)
        )
        out[2] = np.fft.irfftn(
            d_conj[0] * sh_xz + d_conj[1] * sh_yz + d_conj[2] * sh_zz, s=shape_zyx, axes=(0, 1, 2)
        )
        return out

    def _matvec(u_flat: NDArray[np.float64]) -> NDArray[np.float64]:
        u_xyz = u_flat.reshape((3, *shape_zyx))
        eps = _strain_from_u(u_xyz)
        sig = _constitutive_stress(*eps)
        return _div_adjoint(*sig).ravel()

    def _precond(r_flat: NDArray[np.float64]) -> NDArray[np.float64]:
        r_xyz = r_flat.reshape((3, *shape_zyx))
        rh0 = np.fft.rfftn(r_xyz[0], s=shape_zyx, axes=(0, 1, 2))
        rh1 = np.fft.rfftn(r_xyz[1], s=shape_zyx, axes=(0, 1, 2))
        rh2 = np.fft.rfftn(r_xyz[2], s=shape_zyx, axes=(0, 1, 2))

        # Sherman-Morrison inverse of K_0(k) = mu0 |d|^2 I + (lam0 + mu0) d d^*
        d_dot_r = d_conj[0] * rh0 + d_conj[1] * rh1 + d_conj[2] * rh2
        proj = (alpha_sherman * inv_d2) * d_dot_r
        zh0 = inv_mu0 * inv_d2 * (rh0 - d_vec[0] * proj)
        zh1 = inv_mu0 * inv_d2 * (rh1 - d_vec[1] * proj)
        zh2 = inv_mu0 * inv_d2 * (rh2 - d_vec[2] * proj)

        z_out = np.empty((3, *shape_zyx), dtype=np.float64)
        z_out[0] = np.fft.irfftn(zh0, s=shape_zyx, axes=(0, 1, 2))
        z_out[1] = np.fft.irfftn(zh1, s=shape_zyx, axes=(0, 1, 2))
        z_out[2] = np.fft.irfftn(zh2, s=shape_zyx, axes=(0, 1, 2))
        return z_out.ravel()

    lin_op = LinearOperator((n_dof, n_dof), matvec=_matvec, dtype=np.float64)
    pre_op = LinearOperator((n_dof, n_dof), matvec=_precond, dtype=np.float64)

    n_loads = 6 if full_6x6 else 3
    c_cols = np.zeros((6, n_loads), dtype=np.float64)
    iters_list: list[int] = []
    max_rel_res = 0.0
    all_conv = True

    zero_f = np.zeros(shape_zyx, dtype=np.float64)
    one_f = np.ones(shape_zyx, dtype=np.float64)

    for load_idx in range(n_loads):
        macro = [zero_f, zero_f, zero_f, zero_f, zero_f, zero_f]
        macro[load_idx] = one_f
        sig0 = _constitutive_stress(*macro)
        rhs = -_div_adjoint(*sig0).ravel()
        rhs_norm = float(np.linalg.norm(rhs))

        if rhs_norm <= 1e-15:
            u_sol = np.zeros(n_dof, dtype=np.float64)
            iters_list.append(0)
            rel_res = 0.0
        else:
            it_counter = [0]

            def _cb(_xk: NDArray[np.float64]) -> None:
                it_counter[0] += 1

            u_sol, info = cg(
                lin_op,
                rhs,
                rtol=float(rtol),
                atol=0.0,
                maxiter=int(max_iter),
                M=pre_op,
                callback=_cb,
            )
            res_vec = rhs - _matvec(u_sol)
            rel_res = float(np.linalg.norm(res_vec) / rhs_norm)
            max_rel_res = max(max_rel_res, rel_res)
            iters_list.append(int(it_counter[0]))
            if info != 0 and rel_res > float(rtol) * 10.0:
                all_conv = False

        fluct = _strain_from_u(u_sol.reshape((3, *shape_zyx)))
        tot_eps = tuple(macro[m] + fluct[m] for m in range(6))
        sig_tot = _constitutive_stress(*tot_eps)  # type: ignore[arg-type]
        for comp in range(6):
            c_cols[comp, load_idx] = float(np.mean(sig_tot[comp])) * float(youngs_modulus_pa)

    c_norm = 0.5 * (c_cols[:3, :3] + c_cols[:3, :3].T)
    c_hom = 0.5 * (c_cols + c_cols.T) if full_6x6 else None
    e_xyz = uniaxial_moduli_from_normal_stiffness(c_norm)
    m_xyz = (max(0.0, float(c_norm[0, 0])), max(0.0, float(c_norm[1, 1])), max(0.0, float(c_norm[2, 2])))

    return FFTElasticityResult(
        c_norm_3x3_pa=c_norm,
        c_hom_6x6_pa=c_hom,
        e_eff_xyz_pa=e_xyz,
        m_confined_xyz_pa=m_xyz,
        solid_fraction=solid_frac,
        converged=all_conv,
        iterations_per_load=tuple(iters_list),
        max_relative_residual=max_rel_res,
        wall_time_s=time.perf_counter() - t0,
    )


def effective_modulus_fft(
    solid_or_volume: NDArray[Any] | Volume,
    *,
    direction: str = "z",
    confined: bool = False,
    youngs_modulus: float = 10.0e9,
    poisson_ratio: float = 0.25,
    solid_phase: int | Sequence[int] | None = 1,
    void_contrast: float = 0.0,
    rtol: float = 1e-6,
    max_iter: int = 500,
) -> tuple[float, FFTElasticityResult]:
    """Compute directional homogenized stiffness [Pa] matching ``sms.effective_modulus``."""
    d = direction.lower()
    idx = {"x": 0, "y": 1, "z": 2}.get(d)
    if idx is None:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")
    res = homogenize_elastic_fft(
        solid_or_volume,
        youngs_modulus_pa=youngs_modulus,
        poisson_ratio=poisson_ratio,
        solid_phase=solid_phase,
        void_contrast=void_contrast,
        full_6x6=False,
        rtol=rtol,
        max_iter=max_iter,
    )
    val = res.m_confined_xyz_pa[idx] if confined else res.e_eff_xyz_pa[idx]
    return float(val), res
