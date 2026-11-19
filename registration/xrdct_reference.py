"""Stage 0A (Part 1): XRD-CT conjugate COR solver, NNLS phase unmixing, and FRC noise-floor estimation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter, shift as nd_shift
from scipy.optimize import minimize_scalar, nnls
from skimage.transform import iradon

from multimodal_benchmark.io import (
    PowderXRDSpectrum,
    aggregate_peak_maps,
    load_ascii_xrd,
    load_xrdct_peak_maps,
)


@dataclass(frozen=True)
class CORResult:
    center_px: float
    nominal_center_px: float
    offset_from_nominal_px: float
    conjugate_asymmetry: float
    num_conjugate_pairs: int


@dataclass(frozen=True)
class FRCResult:
    frequencies_cyc_per_px: np.ndarray
    frc_curve: np.ndarray
    cutoff_0p5_cyc_per_px: float
    cutoff_0p143_cyc_per_px: float
    effective_resolution_px_0p5: float
    cutoff_full_sb_0p5_cyc_per_px: float = 0.0
    cutoff_half_bit_cyc_per_px: float = 0.0
    effective_resolution_px_full_sb_0p5: float = float("inf")


def estimate_conjugate_cor(
    sinogram_at: np.ndarray,
    angles_deg: np.ndarray,
    max_angle_tol_deg: float = 6.0,
) -> CORResult:
    """Estimate sub-pixel Center of Rotation (COR) from 0-360 deg conjugate projections.

    Handles both uniform 80/40-angle scans and missing-projection scans (e.g. k11-43322 with 79 angles)
    by matching each projection theta_i to its closest conjugate theta_j ~ (theta_i + 180) mod 360.
    For an array of length N (indices 0..N-1), mirror reflection about center c maps x -> 2*c - x.
    """
    sino = np.asarray(sinogram_at, dtype=np.float64)
    angles = np.asarray(angles_deg, dtype=np.float64) % 360.0
    n_ang, n_trans = sino.shape
    mirror_pivot = 0.5 * (n_trans - 1)
    nominal_fbp = float(n_trans // 2)

    pairs: List[Tuple[int, int]] = []
    used = set()
    for i in range(n_ang):
        if i in used:
            continue
        target = (angles[i] + 180.0) % 360.0
        diffs = np.abs(((angles - target + 180.0) % 360.0) - 180.0)
        diffs[i] = 999.0
        j = int(np.argmin(diffs))
        if diffs[j] <= max_angle_tol_deg and j not in used:
            pairs.append((i, j))
            used.add(i)
            used.add(j)

    if not pairs:
        return CORResult(
            center_px=nominal_fbp,
            nominal_center_px=nominal_fbp,
            offset_from_nominal_px=0.0,
            conjugate_asymmetry=float("nan"),
            num_conjugate_pairs=0,
        )

    p_idx = [p[0] for p in pairs]
    q_idx = [p[1] for p in pairs]
    prof_a = sino[p_idx]                    # (P, N)
    prof_b_rev = sino[q_idx, ::-1]          # mirrored around array midpoint (N-1)/2

    def _loss(delta_from_pivot: float) -> float:
        # Shifting mirrored profile by +2*delta reflects about c = mirror_pivot + delta
        shifted = nd_shift(prof_b_rev, shift=(0.0, 2.0 * delta_from_pivot), order=3, mode="nearest")
        margin = max(1, int(np.ceil(abs(2.0 * delta_from_pivot))) + 1)
        if 2 * margin >= n_trans:
            diff = prof_a - shifted
        else:
            diff = prof_a[:, margin:-margin] - shifted[:, margin:-margin]
        return float(np.mean(diff * diff))

    res = minimize_scalar(_loss, bounds=(-0.25 * n_trans, 0.25 * n_trans), method="bounded")
    delta_pivot = float(res.x)
    c_opt = mirror_pivot + delta_pivot

    shifted_opt = nd_shift(prof_b_rev, shift=(0.0, 2.0 * delta_pivot), order=3, mode="nearest")
    denom = float(np.linalg.norm(prof_a) + np.linalg.norm(shifted_opt)) * 0.5 + 1e-12
    asym = float(np.linalg.norm(prof_a - shifted_opt) / denom)

    return CORResult(
        center_px=c_opt,
        nominal_center_px=nominal_fbp,
        offset_from_nominal_px=c_opt - nominal_fbp,
        conjugate_asymmetry=asym,
        num_conjugate_pairs=len(pairs),
    )


def reconstruct_sinogram_2d(
    sinogram_at: np.ndarray,
    angles_deg: np.ndarray,
    center_px: Optional[float] = None,
    filter_name: str = "hann",
    circle: bool = True,
) -> np.ndarray:
    """Reconstruct a 2D slice from a (n_angles, n_translations) sinogram with sub-pixel COR correction."""
    sino = np.asarray(sinogram_at, dtype=np.float64)
    angles = np.asarray(angles_deg, dtype=np.float64)
    # DIAD scans k11-43323..43325 (0..360) and k11-43336..43401 (-47.5..312.5) include a duplicate
    # endpoint frame at theta[0] + 360 deg; drop the duplicate frame so FBP angular weights stay uniform.
    if angles.size > 2 and abs(abs(angles[-1] - angles[0]) - 360.0) < 0.25:
        sino = sino[:-1]
        angles = angles[:-1]
    n_ang, n_trans = sino.shape
    nominal_fbp = float(n_trans // 2)
    if center_px is not None:
        delta = float(center_px) - nominal_fbp
        if abs(delta) > 1e-6:
            sino = nd_shift(sino, shift=(0.0, -delta), order=3, mode="nearest")

    # skimage.transform.iradon expects shape (n_translations, n_angles) and assumes 0..180 scaling;
    # for 0..360 full-turn scans with M projections, divide by 2 so integrated intensity matches 0..180.
    span = float(np.ptp(angles)) if n_ang > 1 else 180.0
    scale = 0.5 if span > 200.0 else 1.0
    rec = iradon(
        sino.T,
        theta=angles,
        output_size=n_trans,
        filter_name=filter_name,
        circle=circle,
        preserve_range=True,
    )
    return (rec * scale).astype(np.float32)


def calibrate_powder_q_shift(
    sinogram_qat: np.ndarray,
    q_grid: np.ndarray,
    na_ref: PowderXRDSpectrum,
    zn_ref: PowderXRDSpectrum,
    q_bounds: Tuple[float, float] = (0.90, 3.05),
    search_range_inv_a: Tuple[float, float] = (-0.004, 0.004),
    num_steps: int = 33,
    broaden_sigma_bins: float = 1.0,
    continuum_window_bins: int = 45,
) -> float:
    """Calibrate zero-point q-shift (A^-1) between laboratory Cu-Ka powder XRD and synchrotron DIAD q-grid."""
    q = np.asarray(q_grid, dtype=np.float64)
    mask = (q >= q_bounds[0]) & (q <= q_bounds[1])
    sub = np.asarray(sinogram_qat[mask], dtype=np.float64)
    mean_ray = np.mean(sub, axis=(1, 2))
    mean_bragg = np.clip(mean_ray - percentile_filter(mean_ray, percentile=15, size=continuum_window_bins), 0.0, None)

    best_dq = 0.0
    best_res = float("inf")
    for dq in np.linspace(search_range_inv_a[0], search_range_inv_a[1], num_steps):
        basis, _ = build_endmember_basis_on_q(
            q,
            na_ref,
            zn_ref,
            cap_ref=None,
            q_bounds=q_bounds,
            q_shift_inv_a=float(dq),
            broaden_sigma_bins=broaden_sigma_bins,
            continuum_window_bins=continuum_window_bins,
        )
        _, rnorm = nnls(basis, mean_bragg)
        if rnorm < best_res:
            best_res = float(rnorm)
            best_dq = float(dq)
    return best_dq


def build_endmember_basis_on_q(
    q_grid: np.ndarray,
    na_ref: PowderXRDSpectrum,
    zn_ref: PowderXRDSpectrum,
    cap_ref: Optional[PowderXRDSpectrum] = None,
    q_bounds: Tuple[float, float] = (0.85, 3.10),
    q_shift_inv_a: float = -0.00125,
    broaden_sigma_bins: float = 1.0,
    continuum_window_bins: int = 45,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate powder XRD endmembers onto DIAD q-grid with zero-point shift, broadening, and continuum removal."""
    q = np.asarray(q_grid, dtype=np.float64)
    mask = (q >= q_bounds[0]) & (q <= q_bounds[1])
    q_sub = q[mask]

    def _interp_bragg(spec: PowderXRDSpectrum) -> np.ndarray:
        # Positive q_shift_inv_a shifts powder peaks to higher q: evaluate powder spectrum at (q_sub - q_shift_inv_a)
        y = np.interp(q_sub - q_shift_inv_a, spec.q_inv_angstrom, spec.intensity, left=0.0, right=0.0)
        if continuum_window_bins > 1:
            y = np.clip(y - percentile_filter(y, percentile=15, size=continuum_window_bins), 0.0, None)
        else:
            y = np.clip(y - np.percentile(y, 5.0), 0.0, None)
        if broaden_sigma_bins > 0.0:
            y = gaussian_filter1d(y, sigma=broaden_sigma_bins)
        return y / (float(np.linalg.norm(y)) + 1e-12)

    na_vec = _interp_bragg(na_ref)
    zn_vec = _interp_bragg(zn_ref)

    cols = [na_vec, zn_vec]
    if cap_ref is not None:
        cap_raw = np.interp(q_sub - q_shift_inv_a, cap_ref.q_inv_angstrom, cap_ref.intensity, left=0.0, right=0.0)
        cap_raw = np.clip(cap_raw - np.percentile(cap_raw, 2.0), 0.0, None)
        cols.append(cap_raw / (float(np.linalg.norm(cap_raw)) + 1e-12))
    basis = np.column_stack(cols)
    return basis, mask


def unmix_sinogram_nnls(
    sinogram_qat: np.ndarray,
    q_grid: np.ndarray,
    na_basis_or_spec: np.ndarray | PowderXRDSpectrum,
    zn_basis_or_spec: np.ndarray | PowderXRDSpectrum,
    cap_basis_or_spec: Optional[np.ndarray | PowderXRDSpectrum] = None,
    q_bounds: Tuple[float, float] = (0.85, 3.10),
    subtract_continuum: Optional[bool] = None,
    q_shift_inv_a: float = -0.00125,
    broaden_sigma_bins: float = 1.0,
    continuum_window_bins: int = 45,
) -> Dict[str, np.ndarray]:
    """Unmix a (n_q, n_angles, n_trans) XRD-CT sinogram into Na-13X, Zn-13X, and continuum/capillary sinograms.

    When laboratory PowderXRDSpectrum references are supplied, per-ray rolling continuum subtraction removes the
    air + borosilicate glass scatter baseline prior to sharp Faujasite Bragg NNLS unmixing, eliminating both the
    outer capillary halo leak into zeolite maps and cross-phase doublet wing spillover.
    """
    q = np.asarray(q_grid, dtype=np.float64)
    is_powder_spec = isinstance(na_basis_or_spec, PowderXRDSpectrum)
    if subtract_continuum is None:
        subtract_continuum = is_powder_spec

    if is_powder_spec:
        assert isinstance(zn_basis_or_spec, PowderXRDSpectrum)
        # If per-ray continuum is removed first, fit pure 2-phase Bragg [Na-13X, Zn-13X] without broad glass leakage
        cap_spec = (
            cap_basis_or_spec
            if (not subtract_continuum and isinstance(cap_basis_or_spec, PowderXRDSpectrum))
            else None
        )
        basis, mask = build_endmember_basis_on_q(
            q,
            na_basis_or_spec,
            zn_basis_or_spec,
            cap_spec,
            q_bounds=q_bounds,
            q_shift_inv_a=q_shift_inv_a,
            broaden_sigma_bins=broaden_sigma_bins,
            continuum_window_bins=continuum_window_bins if subtract_continuum else 0,
        )
    else:
        mask = (q >= q_bounds[0]) & (q <= q_bounds[1])
        cols = [
            np.asarray(na_basis_or_spec, dtype=np.float64)[mask],
            np.asarray(zn_basis_or_spec, dtype=np.float64)[mask],
        ]
        if cap_basis_or_spec is not None and not subtract_continuum:
            cols.append(np.asarray(cap_basis_or_spec, dtype=np.float64)[mask])
        cols = [c / (np.linalg.norm(c) + 1e-12) for c in cols]
        basis = np.column_stack(cols)

    sub = np.asarray(sinogram_qat[mask], dtype=np.float64)
    continuum_sino: Optional[np.ndarray] = None
    if subtract_continuum:
        bg_qat = percentile_filter(sub, percentile=15, size=(continuum_window_bins, 1, 1))
        # Subtract air baseline floor (outermost rays) so capillary sinogram isolates the glass wall + pellet scatter
        bg_mean = np.mean(bg_qat, axis=0)
        air_floor = float(np.percentile(bg_mean, 10.0))
        continuum_sino = np.clip(bg_mean - air_floor, 0.0, None).astype(np.float32)
        sub = np.clip(sub - bg_qat, 0.0, None)

    _, n_ang, n_trans = sub.shape
    n_comp = basis.shape[1]
    coeffs = np.zeros((n_comp, n_ang, n_trans), dtype=np.float32)

    for ia in range(n_ang):
        for it in range(n_trans):
            spectrum = np.clip(sub[:, ia, it], 0.0, None)
            sol, _ = nnls(basis, spectrum)
            coeffs[:, ia, it] = sol.astype(np.float32)

    out: Dict[str, np.ndarray] = {"na": coeffs[0], "zn": coeffs[1]}
    if continuum_sino is not None:
        out["continuum"] = continuum_sino
        if cap_basis_or_spec is not None:
            out["capillary"] = continuum_sino
    elif n_comp >= 3:
        out["capillary"] = coeffs[2]
    return out


def legacy_two_bin_window_sinogram(
    sinogram_qat: np.ndarray,
    q_grid: np.ndarray,
    na_q: float = 1.650,
    zn_q: float = 1.655,
) -> Dict[str, np.ndarray]:
    """Reproduce the legacy Sinograms.ipynb asymmetric 2-bin window [idx-1:idx+1] for diagnostic comparison."""
    q = np.asarray(q_grid, dtype=np.float64)
    na_idx = int(np.argmin(np.abs(q - na_q)))
    zn_idx = int(np.argmin(np.abs(q - zn_q)))
    na_sino = np.sum(sinogram_qat[max(0, na_idx - 1) : na_idx + 1], axis=0).astype(np.float32)
    zn_sino = np.sum(sinogram_qat[max(0, zn_idx - 1) : zn_idx + 1], axis=0).astype(np.float32)
    return {"na": na_sino, "zn": zn_sino}


def compute_2d_frc(img_a: np.ndarray, img_b: np.ndarray) -> FRCResult:
    """Compute 2D Fourier Ring Correlation (FRC) between two independent half-maps."""
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Expected matching 2D arrays, got {a.shape} and {b.shape}")

    ny, nx = a.shape
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    win = np.outer(wy, wx)

    fa = np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))
    fb = np.fft.fftshift(np.fft.fft2((b - b.mean()) * win))

    cy, cx = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    radii = np.rint(np.hypot(yy - cy, xx - cx)).astype(np.int32)
    max_r = min(cy, cx)

    freqs: List[float] = []
    frc_vals: List[float] = []
    half_bit_vals: List[float] = []
    num_prod = np.real(fa * np.conj(fb))
    pow_a = np.abs(fa) ** 2
    pow_b = np.abs(fb) ** 2

    for r in range(1, max_r + 1):
        ring = radii == r
        nr = int(np.count_nonzero(ring))
        if nr < 4:
            continue
        denom = np.sqrt(np.sum(pow_a[ring]) * np.sum(pow_b[ring])) + 1e-15
        frc_vals.append(float(np.sum(num_prod[ring]) / denom))
        freqs.append(float(r) / float(min(ny, nx)))
        half_bit_vals.append(float((0.2071 + 1.9102 / np.sqrt(nr)) / (1.2071 + 0.9102 / np.sqrt(nr))))

    f_arr = np.asarray(freqs, dtype=np.float64)
    c_arr = np.asarray(frc_vals, dtype=np.float64)
    hb_arr = np.asarray(half_bit_vals, dtype=np.float64)
    # Spearman-Brown full-dataset correction (Rosenthal & Henderson 2003): FRC_full = 2*FRC_half / (1 + FRC_half)
    sb_arr = np.where(c_arr > 0.0, (2.0 * c_arr) / (1.0 + np.maximum(c_arr, -0.99)), c_arr)

    def _first_crossing(curve: np.ndarray, thresh: float | np.ndarray) -> float:
        if f_arr.size == 0:
            return 0.0
        t_vec = np.full_like(curve, thresh) if np.ndim(thresh) == 0 else np.asarray(thresh, dtype=np.float64)
        diff = curve - t_vec
        below = np.where(diff < 0.0)[0]
        if below.size == 0:
            return float(f_arr[-1])
        idx = int(below[0])
        if idx == 0:
            return float(f_arr[0])
        f0, f1 = f_arr[idx - 1], f_arr[idx]
        t = (0.0 - diff[idx - 1]) / (diff[idx] - diff[idx - 1] - 1e-12)
        return float(f0 + np.clip(t, 0.0, 1.0) * (f1 - f0))

    cut_05 = _first_crossing(c_arr, 0.5)
    cut_0143 = _first_crossing(c_arr, 0.143)
    cut_sb05 = _first_crossing(sb_arr, 0.5)
    cut_hb = _first_crossing(c_arr, hb_arr)
    eff_res = (1.0 / cut_05) if cut_05 > 1e-6 else float("inf")
    eff_res_sb = (1.0 / cut_sb05) if cut_sb05 > 1e-6 else float("inf")
    return FRCResult(
        frequencies_cyc_per_px=f_arr,
        frc_curve=c_arr,
        cutoff_0p5_cyc_per_px=cut_05,
        cutoff_0p143_cyc_per_px=cut_0143,
        effective_resolution_px_0p5=eff_res,
        cutoff_full_sb_0p5_cyc_per_px=cut_sb05,
        cutoff_half_bit_cyc_per_px=cut_hb,
        effective_resolution_px_full_sb_0p5=eff_res_sb,
    )


def estimate_odd_even_noise_floor(
    sinogram_at: np.ndarray,
    angles_deg: np.ndarray,
    center_px: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, FRCResult]:
    """Reconstruct odd and even angular projections independently to estimate full-map variance sigma^2_ref(x,y)."""
    sino = np.asarray(sinogram_at, dtype=np.float64)
    ang = np.asarray(angles_deg, dtype=np.float64)
    if ang.size > 2 and abs(abs(float(ang[-1] - ang[0])) - 360.0) < 0.35:
        sino = sino[:-1]
        ang = ang[:-1]
    rec_full = reconstruct_sinogram_2d(sino, ang, center_px=center_px)
    rec_even = reconstruct_sinogram_2d(sino[0::2], ang[0::2], center_px=center_px, circle=False)
    rec_odd = reconstruct_sinogram_2d(sino[1::2], ang[1::2], center_px=center_px, circle=False)

    # Full map is average of even and odd half-maps -> Var(0.5*(E+O)) = 0.25 * (E - O)^2
    raw_var = 0.25 * (rec_even.astype(np.float64) - rec_odd.astype(np.float64)) ** 2
    floor = float(np.percentile(raw_var, 25.0)) * 0.25 + 1e-8
    sigma2 = np.maximum(raw_var, floor).astype(np.float32)
    frc = compute_2d_frc(rec_even, rec_odd)
    return rec_full, sigma2, frc


def estimate_reflection_subset_noise_floor(peaks: Dict[float, np.ndarray]) -> Dict[str, Any]:
    """Estimate multi-reflection reference map and pixelwise noise variance from disjoint low-q vs high-q subsets."""
    full_map = aggregate_peak_maps(peaks, mode="all_sum")
    low_map = aggregate_peak_maps(peaks, mode="low_q")
    high_map = aggregate_peak_maps(peaks, mode="high_q")

    # Rescale low/high subsets to have matching total signal before differencing
    scale_high = float(np.sum(np.abs(low_map)) / (np.sum(np.abs(high_map)) + 1e-12))
    high_scaled = high_map * scale_high
    # Summed map S = L + H; using scaled difference delta = L - alpha*H to bound variance
    diff = low_map.astype(np.float64) - high_scaled.astype(np.float64)
    sigma2 = np.maximum(diff * diff, float(np.percentile(diff * diff, 25.0)) * 0.25 + 1e-6).astype(np.float32)
    frc = compute_2d_frc(low_map, high_scaled)
    corr = float(np.corrcoef(low_map.ravel(), high_map.ravel())[0, 1])

    return {
        "map": full_map,
        "sigma2": sigma2,
        "low_high_pearson_r": corr,
        "frc": frc,
    }


def audit_cached_xrdct_outputs(
    data_dir: Path | str,
    out_dir: Optional[Path | str] = None,
    scans_25um: Sequence[int] = (43322, 43323, 43324, 43325),
) -> Dict[str, Any]:
    """Audit real Zenodo powder XRD (.asc) and 25 um / 50 um XRD-CT peak maps in data_dir."""
    ddir = Path(data_dir)
    report: Dict[str, Any] = {"data_dir": str(ddir), "powder_xrd": {}, "scans_25um": {}}

    # 1. Powder XRD endmember check if .asc files exist
    na_asc = ddir / "13Xhighres_2-Theta_Omega.asc"
    zn_asc = ddir / "Zn13Xhighres_2-Theta_Omega.asc"
    cap_asc = ddir / "empty_cap_2-Theta_Omega.asc"
    if na_asc.exists() and zn_asc.exists():
        na_spec = load_ascii_xrd(na_asc)
        zn_spec = load_ascii_xrd(zn_asc)
        cap_spec = load_ascii_xrd(cap_asc) if cap_asc.exists() else None
        win_na = (na_spec.q_inv_angstrom >= 1.62) & (na_spec.q_inv_angstrom <= 1.69)
        win_zn = (zn_spec.q_inv_angstrom >= 1.62) & (zn_spec.q_inv_angstrom <= 1.69)
        q_na_peak = float(na_spec.q_inv_angstrom[win_na][np.argmax(na_spec.intensity[win_na])])
        q_zn_peak = float(zn_spec.q_inv_angstrom[win_zn][np.argmax(zn_spec.intensity[win_zn])])
        report["powder_xrd"] = {
            "num_points": int(na_spec.q_inv_angstrom.size),
            "na_doublet_peak_q_inv_A": round(q_na_peak, 5),
            "zn_doublet_peak_q_inv_A": round(q_zn_peak, 5),
            "peak_shift_inv_A": round(q_zn_peak - q_na_peak, 5),
            "empty_capillary_loaded": cap_spec is not None,
        }

    # 2. Audit 25 um XRD-CT reference slices
    if out_dir is not None:
        odir = Path(out_dir)
        odir.mkdir(parents=True, exist_ok=True)
    else:
        odir = None

    for sid in scans_25um:
        f_na = ddir / f"output_{sid}_na.nxs"
        f_zn = ddir / f"output_{sid}_zn.nxs"
        if not (f_na.exists() and f_zn.exists()):
            continue
        peaks_na = load_xrdct_peak_maps(f_na)
        peaks_zn = load_xrdct_peak_maps(f_zn)
        est_na = estimate_reflection_subset_noise_floor(peaks_na)
        est_zn = estimate_reflection_subset_noise_floor(peaks_zn)

        na_single = aggregate_peak_maps(peaks_na, mode="single_peak", target_q=1.651)
        zn_single = aggregate_peak_maps(peaks_zn, mode="single_peak", target_q=1.656)
        crosstalk_single = float(np.corrcoef(na_single.ravel(), zn_single.ravel())[0, 1])
        crosstalk_multi = float(np.corrcoef(est_na["map"].ravel(), est_zn["map"].ravel())[0, 1])

        report["scans_25um"][str(sid)] = {
            "num_na_reflections": len(peaks_na),
            "num_zn_reflections": len(peaks_zn),
            "na_low_vs_high_q_r": round(est_na["low_high_pearson_r"], 4),
            "zn_low_vs_high_q_r": round(est_zn["low_high_pearson_r"], 4),
            "na_frc_cutoff_0p5": round(est_na["frc"].cutoff_0p5_cyc_per_px, 4),
            "zn_frc_cutoff_0p5": round(est_zn["frc"].cutoff_0p5_cyc_per_px, 4),
            "na_zn_corr_single_1p65_peak": round(crosstalk_single, 4),
            "na_zn_corr_all_peaks": round(crosstalk_multi, 4),
        }

        if odir is not None:
            np.savez_compressed(
                odir / f"ref_25um_{sid}.npz",
                na=est_na["map"],
                zn=est_zn["map"],
                na_sigma2=est_na["sigma2"],
                zn_sigma2=est_zn["sigma2"],
            )

    if odir is not None:
        with open(odir / "xrdct_audit_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 0A XRD-CT reference & noise-floor audit.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cache/zenodo"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/stage0_xrdct"))
    args = parser.parse_args(argv)
    report = audit_cached_xrdct_outputs(args.data_dir, args.out_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
