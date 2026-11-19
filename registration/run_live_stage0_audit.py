"""Live Stage 0A & Stage 0B empirical audit runner on Zenodo XRD-CT sinograms and I13-2 XCT sub-volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from skimage.transform import downscale_local_mean

from baselines.classical import lanczos_upsample_3d, tricubic_upsample_3d, tv_deconv_upsample_3d
from multimodal_benchmark.evaluator import evaluate_bundles
from multimodal_benchmark.io import load_ascii_xrd, load_xrdct_sinogram
from registration.i13_ladder import register_3d_similarity, resample_moving_to_fixed_zyx
from registration.xrdct_reference import (
    estimate_conjugate_cor,
    estimate_odd_even_noise_floor,
    reconstruct_sinogram_2d,
    unmix_sinogram_nnls,
)


def _inspect_nexus_stage_motors(nxs_path: Path) -> Dict[str, Any]:
    """Extract scalar/mean stage motor coordinates (theta, y, z, x) from a DIAD NeXus sinogram file."""
    info: Dict[str, Any] = {}
    with h5py.File(nxs_path, "r") as h5f:
        def _visit(name: str, obj: object) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            lname = name.lower()
            if any(k in lname for k in ("gts_cs_", "ss2_", "stage", "samy", "samz", "kb_")):
                if obj.size <= 200:
                    arr = np.asarray(obj[:])
                    if np.issubdtype(arr.dtype, np.number):
                        if arr.size == 1:
                            info[name] = float(arr.ravel()[0])
                        else:
                            info[name] = {
                                "shape": list(arr.shape),
                                "min": round(float(np.min(arr)), 4),
                                "max": round(float(np.max(arr)), 4),
                                "mean": round(float(np.mean(arr)), 4),
                            }
        h5f.visititems(_visit)
    return info


def audit_live_xrdct_sinograms(
    cache_dir: Path,
    fig_out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit real DIAD 25 um (43322-43325) and sample 50 um sinograms: motors, COR, odd/even FRC, and NNLS."""
    na_spec = load_ascii_xrd(cache_dir / "13Xhighres_2-Theta_Omega.asc")
    zn_spec = load_ascii_xrd(cache_dir / "Zn13Xhighres_2-Theta_Omega.asc")
    cap_spec = load_ascii_xrd(cache_dir / "empty_cap_2-Theta_Omega.asc")

    results: Dict[str, Any] = {"scans_25um": {}, "scans_50um_sample": {}}

    sino_files_25 = [
        (43322, "k11-43322-diffraction-DiffInt_Mask_Azz_Txt.nxs"),
        (43323, "k11-43323-diffraction-DiffInt_Mask_Azz_Txt.nxs"),
        (43324, "k11-43324-diffraction-DiffInt_Mask_Azz_Txt.nxs"),
        (43325, "k11-43325-diffraction-DiffInt_Mask_Azz_Txt.nxs"),
    ]
    nnls_examples: Dict[int, Dict[str, np.ndarray]] = {}

    for sid, fname in sino_files_25:
        fpath = cache_dir / fname
        if not fpath.exists():
            continue
        motors = _inspect_nexus_stage_motors(fpath)
        sino_obj = load_xrdct_sinogram(fpath, q_range=(0.85, 3.10), scan_id=sid)
        total_sino = np.sum(np.clip(sino_obj.data, 0.0, None), axis=0)  # (n_angles, n_trans)

        cor = estimate_conjugate_cor(total_sino, sino_obj.angles_deg)
        rec_tot, sigma2_tot, frc_tot = estimate_odd_even_noise_floor(
            total_sino,
            sino_obj.angles_deg,
            center_px=cor.center_px,
        )

        # Multi-peak NNLS unmixing with empty-capillary background subtraction
        unmixed = unmix_sinogram_nnls(
            sino_obj.data,
            sino_obj.q_inv_angstrom,
            na_spec,
            zn_spec,
            cap_spec,
            q_bounds=(0.90, 3.05),
        )
        na_rec, na_oe_s2, na_frc = estimate_odd_even_noise_floor(
            unmixed["na"], sino_obj.angles_deg, center_px=cor.center_px
        )
        zn_rec, zn_oe_s2, zn_frc = estimate_odd_even_noise_floor(
            unmixed["zn"], sino_obj.angles_deg, center_px=cor.center_px
        )
        cap_rec = reconstruct_sinogram_2d(unmixed["capillary"], sino_obj.angles_deg, center_px=cor.center_px)

        nnls_corr = float(np.corrcoef(na_rec.ravel(), zn_rec.ravel())[0, 1])
        nnls_examples[sid] = {
            "na": na_rec,
            "zn": zn_rec,
            "capillary": cap_rec,
            "zn_oe_sigma": np.sqrt(zn_oe_s2),
            "frc_freqs": zn_frc.frequencies_cyc_per_px,
            "frc_zn": zn_frc.frc_curve,
            "frc_na": na_frc.frc_curve,
        }

        results["scans_25um"][str(sid)] = {
            "sinogram_shape_qat": list(sino_obj.data.shape),
            "angles_min_max_deg": [round(float(sino_obj.angles_deg.min()), 3), round(float(sino_obj.angles_deg.max()), 3)],
            "conjugate_cor_px": round(cor.center_px, 4),
            "nominal_cor_px": cor.nominal_center_px,
            "cor_offset_px": round(cor.offset_from_nominal_px, 4),
            "cor_offset_um": round(cor.offset_from_nominal_px * 25.0, 2),
            "conjugate_self_absorption_asymmetry": round(cor.conjugate_asymmetry, 4),
            "total_odd_even_frc_0p5_cyc_px": round(frc_tot.cutoff_0p5_cyc_per_px, 4),
            "na_odd_even_frc_0p5_cyc_px": round(na_frc.cutoff_0p5_cyc_per_px, 4),
            "zn_odd_even_frc_0p5_cyc_px": round(zn_frc.cutoff_0p5_cyc_per_px, 4),
            "zn_effective_res_um_at_frc0p5": round(zn_frc.effective_resolution_px_0p5 * 25.0, 2),
            "nnls_na_zn_spatial_corr": round(nnls_corr, 4),
            "motors": motors,
        }

    for sid in (43336, 43345, 43393, 43401):
        fpath = cache_dir / f"k11-{sid}-diffraction-DiffInt_Mask_Azz_Txt_50um.nxs"
        if not fpath.exists():
            continue
        motors = _inspect_nexus_stage_motors(fpath)
        sino_obj = load_xrdct_sinogram(fpath, q_range=(0.85, 3.10), scan_id=sid)
        total_sino = np.sum(np.clip(sino_obj.data, 0.0, None), axis=0)
        cor = estimate_conjugate_cor(total_sino, sino_obj.angles_deg)
        results["scans_50um_sample"][str(sid)] = {
            "sinogram_shape_qat": list(sino_obj.data.shape),
            "angles_min_max_deg": [round(float(sino_obj.angles_deg.min()), 3), round(float(sino_obj.angles_deg.max()), 3)],
            "conjugate_cor_px": round(cor.center_px, 4),
            "cor_offset_px": round(cor.offset_from_nominal_px, 4),
            "cor_offset_um": round(cor.offset_from_nominal_px * 50.0, 2),
            "conjugate_asymmetry": round(cor.conjugate_asymmetry, 4) if np.isfinite(cor.conjugate_asymmetry) else None,
            "motors": motors,
        }

    f25_43324 = cache_dir / "k11-43324-diffraction-DiffInt_Mask_Azz_Txt.nxs"
    f50_43345 = cache_dir / "k11-43345-diffraction-DiffInt_Mask_Azz_Txt_50um.nxs"
    if f25_43324.exists() and f50_43345.exists():
        s25 = load_xrdct_sinogram(f25_43324, q_range=(1.648, 1.658), scan_id=43324)
        s50 = load_xrdct_sinogram(f50_43345, q_range=(1.648, 1.658), scan_id=43345)
        p25 = s25.data.sum(0)[::2, :40].reshape(40, 20, 2).mean(-1)
        p50 = s50.data.sum(0)
        p25_d = p25 - p25.mean(axis=1, keepdims=True)
        p50_d = p50 - p50.mean(axis=1, keepdims=True)
        results["gts_cs_theta_motor_zero_audit"] = {
            "scan_25um_43324_theta_deg": [round(float(s25.angles_deg[0]), 2), round(float(s25.angles_deg[-1]), 2)],
            "scan_50um_43345_theta_deg": [round(float(s50.angles_deg[0]), 2), round(float(s50.angles_deg[-1]), 2)],
            "soft_motor_zero_offset_deg": -47.5,
            "frame0_asymmetric_sinogram_pearson_r": round(float(np.corrcoef(p25_d.ravel(), p50_d.ravel())[0, 1]), 5),
            "kb_cs_x_center_diad_xct_px": round(float(np.mean(s25.kb_cs_x_px)), 2) if s25.kb_cs_x_px is not None else None,
            "kb_cs_x_span_um": round(float((s25.kb_cs_x_px[-1] - s25.kb_cs_x_px[0]) * 0.54), 2) if s25.kb_cs_x_px is not None else None,
        }

    if fig_out_path is not None and 43324 in nnls_examples:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ex = nnls_examples[43324]
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), constrained_layout=True)
        im0 = axes[0].imshow(ex["na"], cmap="Blues")
        axes[0].set_title("NNLS Na-13X (43324, COR-corrected)", fontsize=9.5)
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(ex["zn"], cmap="Oranges")
        axes[1].set_title("NNLS Zn-13X (43324, COR-corrected)", fontsize=9.5)
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(ex["capillary"], cmap="Greys")
        axes[2].set_title("Unmixed Borosilicate Capillary Ring", fontsize=9.5)
        fig.colorbar(im2, ax=axes[2], fraction=0.046)

        axes[3].plot(ex["frc_freqs"], ex["frc_zn"], lw=2.0, color="darkorange", label="Zn-13X Odd/Even FRC")
        axes[3].plot(ex["frc_freqs"], ex["frc_na"], lw=2.0, color="steelblue", label="Na-13X Odd/Even FRC")
        axes[3].axhline(0.5, ls="--", color="black", lw=1.0, label="FRC = 0.5")
        axes[3].axhline(0.143, ls=":", color="gray", lw=1.0, label="FRC = 0.143")
        axes[3].set_ylim(-0.2, 1.05)
        axes[3].set_xlabel("Spatial Frequency (cycles / 25 µm px)")
        axes[3].set_ylabel("FRC")
        axes[3].set_title("Half-Map Projection FRC (Scan 43324)", fontsize=9.5)
        axes[3].legend(fontsize=7.5, loc="lower left")

        for ax in axes[:3]:
            ax.axis("off")
        fig_out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_out_path, dpi=160)
        plt.close(fig)

    return results


def audit_live_i13_8bit_ladder(cache_dir: Path) -> Dict[str, Any]:
    """Inspect I13-2 8-bit cropped sub-volumes (14245751), run 7-DOF 2x registration, and score T1 baselines."""
    out: Dict[str, Any] = {"processed_nxs_cor": {}, "subvolume_shapes": {}, "t1_2x_live_registration_and_baselines": {}}

    for sid, vox in [("169068", 2.6), ("169067", 1.625), ("169066", 0.8125), ("169065", 0.325)]:
        p_nxs = cache_dir / f"{sid}_processed.nxs"
        if p_nxs.exists():
            with h5py.File(p_nxs, "r") as f:
                cor_ds = "entry/final_result_tomo/meta_data/centre_of_rotation/centre_of_rotation"
                if cor_ds in f:
                    cors = np.asarray(f[cor_ds][:], dtype=np.float64)
                    out["processed_nxs_cor"][sid] = {
                        "voxel_um": vox,
                        "cor_mean_px": round(float(np.mean(cors)), 3),
                        "cor_min_max_px": [round(float(np.min(cors)), 3), round(float(np.max(cors)), 3)],
                    }
        p_h5 = cache_dir / f"{sid}_8bit_cropped.h5"
        if p_h5.exists():
            with h5py.File(p_h5, "r") as f:
                k = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)][0]
                ds = f[k]
                out["subvolume_shapes"][sid] = {"dataset": k, "shape": list(ds.shape), "dtype": str(ds.dtype)}

    # Live 2x (169067 1.625 um -> 169066 0.8125 um) coarse-to-fine 7-DOF registration & T1 baseline benchmark
    p_lo = cache_dir / "169067_8bit_cropped.h5"
    p_hi = cache_dir / "169066_8bit_cropped.h5"
    if p_lo.exists() and p_hi.exists():
        from scipy.signal import fftconvolve

        with h5py.File(p_lo, "r") as flo, h5py.File(p_hi, "r") as fhi:
            klo = [k for k in flo.keys() if isinstance(flo[k], h5py.Dataset)][0]
            khi = [k for k in fhi.keys() if isinstance(fhi[k], h5py.Dataset)][0]
            vol_lo = np.asarray(flo[klo][:], dtype=np.float32)
            vol_hi = np.asarray(fhi[khi][:], dtype=np.float32)

        # Stage 1: Global 3D FFT normalized cross-correlation at 6.5 um/voxel (4x on 169067, 8x on 169066)
        lo_coarse = downscale_local_mean(vol_lo[:496, :1000, :1000], (4, 4, 4))
        hi_coarse = downscale_local_mean(vol_hi[:496, :1000, :1000], (8, 8, 8))
        tmpl = (hi_coarse - hi_coarse.mean()) / (hi_coarse.std() + 1e-8)
        lo_zero = lo_coarse - lo_coarse.mean()
        num = fftconvolve(lo_zero, tmpl[::-1, ::-1, ::-1], mode="valid")
        ones = np.ones_like(tmpl)
        lo_sq = fftconvolve(lo_zero ** 2, ones, mode="valid")
        lo_m = fftconvolve(lo_zero, ones, mode="valid") / tmpl.size
        den = np.sqrt(np.maximum(lo_sq - tmpl.size * (lo_m ** 2), 1e-6)) * np.sqrt(tmpl.size)
        ncc_grid = num / den
        bz, by, bx = [int(v) for v in np.unravel_index(np.argmax(ncc_grid), ncc_grid.shape)]
        coarse_ncc_peak = float(ncc_grid[bz, by, bx])

        cz_lo = bz * 4 + (vol_hi.shape[0] // 4)
        cy_lo = by * 4 + (vol_hi.shape[1] // 4)
        cx_lo = bx * 4 + (vol_hi.shape[2] // 4)
        coarse_offset_from_geom_center_um = [
            round((cz_lo - vol_lo.shape[0] // 2) * 1.625, 2),
            round((cy_lo - vol_lo.shape[1] // 2) * 1.625, 2),
            round((cx_lo - vol_lo.shape[2] // 2) * 1.625, 2),
        ]

        geom_lr = vol_lo[250 - 32 : 250 + 32, 500 - 32 : 500 + 32, 500 - 32 : 500 + 32]
        inner_hr_geom = vol_hi[250 - 64 : 250 + 64, 500 - 64 : 500 + 64, 500 - 64 : 500 + 64]
        naive_geom_r = float(np.corrcoef(geom_lr.ravel(), downscale_local_mean(inner_hr_geom, (2, 2, 2)).ravel())[0, 1])

        # Stage 2: Sub-voxel 7-DOF SimpleITK similarity refinement on 96^3 LR <-> 192^3 HR cubes around coarse match
        lr_sz = 96
        hr_sz = 192
        lr_cube = vol_lo[
            cz_lo - lr_sz // 2 : cz_lo + lr_sz // 2,
            cy_lo - lr_sz // 2 : cy_lo + lr_sz // 2,
            cx_lo - lr_sz // 2 : cx_lo + lr_sz // 2,
        ]
        hr_cube = vol_hi[
            250 - hr_sz // 2 : 250 + hr_sz // 2,
            500 - hr_sz // 2 : 500 + hr_sz // 2,
            500 - hr_sz // 2 : 500 + hr_sz // 2,
        ]

        reg = register_3d_similarity(
            fixed_zyx=hr_cube,
            moving_zyx=lr_cube,
            fixed_spacing_zyx_um=(0.8125, 0.8125, 0.8125),
            moving_spacing_zyx_um=(1.625, 1.625, 1.625),
            metric="mattes_mi",
            max_iterations=200,
            shrink_factors=(4, 2, 1),
            smoothing_sigmas=(2.0, 1.0, 0.0),
            allow_scale=True,
        )

        inner_hr = hr_cube[56:136, 56:136, 56:136]  # 80^3 HR (65 um)^3
        inner_origin_um = (56 * 0.8125, 56 * 0.8125, 56 * 0.8125)
        aligned_lr = resample_moving_to_fixed_zyx(
            moving_zyx=lr_cube,
            moving_spacing_zyx_um=(1.625, 1.625, 1.625),
            fixed_shape_zyx=(40, 40, 40),
            fixed_spacing_zyx_um=(1.625, 1.625, 1.625),
            transform=reg.transform,
            fixed_origin_zyx_um=inner_origin_um,
        )

        coarse_only_lr = lr_cube[28:68, 28:68, 28:68]
        hr_down2 = downscale_local_mean(inner_hr, (2, 2, 2))
        coarse_r = float(np.corrcoef(coarse_only_lr.ravel(), hr_down2.ravel())[0, 1])
        aligned_r = float(np.corrcoef(aligned_lr.ravel(), hr_down2.ravel())[0, 1])

        baselines_out: Dict[str, Any] = {}
        for bname, pred in [
            ("Tricubic-3D", tricubic_upsample_3d(aligned_lr, inner_hr.shape)),
            ("Lanczos-3D", lanczos_upsample_3d(aligned_lr, inner_hr.shape)),
            ("TV-Deconv-3D", tv_deconv_upsample_3d(aligned_lr, inner_hr.shape, n_iter=8)),
        ]:
            ev = evaluate_bundles(
                "t1",
                {"pred": pred},
                {"pred": inner_hr},
                method_name=bname,
                voxel_size_um=0.8125,
                with_transport=True,
                bootstrap_samples=100,
            )
            baselines_out[bname] = {k: round(v, 4) for k, v in ev["metrics"].items() if np.isfinite(v)}

        out["t1_2x_live_registration_and_baselines"] = {
            "coarse_fft_ncc_peak": round(coarse_ncc_peak, 4),
            "coarse_center_in_169067_zyx_vox": [cz_lo, cy_lo, cx_lo],
            "coarse_offset_from_geom_center_zyx_um": coarse_offset_from_geom_center_um,
            "scale_factor_fitted": round(reg.transform.scale, 5),
            "fine_residual_translation_xyz_um": [round(v, 3) for v in reg.transform.translation_xyz_um],
            "tre_mean_um": round(reg.tre_mean_um, 3) if np.isfinite(reg.tre_mean_um) else None,
            "tre_median_um": round(reg.tre_median_um, 3) if np.isfinite(reg.tre_median_um) else None,
            "num_matched_landmarks": reg.num_matched_landmarks,
            "pearson_r_naive_geometric_center": round(naive_geom_r, 4),
            "pearson_r_after_coarse_fft": round(coarse_r, 4),
            "pearson_r_after_fine_7dof": round(aligned_r, 4),
            "evaluated_hr_cube_shape": list(inner_hr.shape),
            "classical_baselines": baselines_out,
        }

    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run live Stage 0A/0B audit on cached Zenodo sinograms & XCT sub-volumes.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/zenodo"))
    parser.add_argument("--out-json", type=Path, default=Path("configs/stage0_live_audit_results.json"))
    parser.add_argument("--fig-out", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = {
        "stage0a_xrdct_sinograms": audit_live_xrdct_sinograms(args.cache_dir, fig_out_path=args.fig_out),
        "stage0b_i13_ladder": audit_live_i13_8bit_ladder(args.cache_dir),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
