"""Live empirical runner for XRD-CT classical baselines (Task T2 unguided 4-fold LOSO & Optional Task T4 sparse-view).

Executes strictly on real 32-bit float DIAD XRD-CT sinograms cached from Zenodo `13329639` + Rigaku powder references
from Zenodo `13329670`:
  - Calibrated NNLS phase unmixing (`dq = -0.00125 A^-1`, empty-capillary continuum subtraction, `0/360 deg` duplicate
    frame merge, conjugate COR correction, `+47.5 deg` 50 um motor-zero offset correction).
  - Task T2 (`50 -> 25 um` unguided baselines: Nearest, Bilinear, Bicubic) evaluated under strict 4-fold Leave-One-
    Slice-Out (LOSO) cross-validation (`alpha_fold` exposure ratio fitted ONLY on the 3 training slices per fold),
    reporting both the 3 confirmed-margin heights (`43322, 43323, 43325`) and the 4-slice mean (`43324` flagged
    provisional pending Stage 0A-2, plus explicit `43345` vs `43346` sensitivity).
  - Optional Task T4 (`10`, `20`, and `40` views out of `79` unique projections: `FBP` vs `Non-Negative SIRT`) across
    all four `25 um` scans (`43324, 43322, 43323, 43325`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from baselines.classical import fbp_sparse_reconstruct_2d, sirt_reconstruct_2d
from multimodal_benchmark.evaluator import evaluate_bundles
from multimodal_benchmark.io import load_ascii_xrd, load_xrdct_sinogram
from registration.xrdct_reference import (
    estimate_conjugate_cor,
    estimate_odd_even_noise_floor,
    unmix_sinogram_nnls,
)

PAIRED_25_TO_50: Tuple[Tuple[int, int, int], ...] = (
    (43324, 43345, 9),
    (43322, 43395, 14),
    (43323, 43398, 17),
    (43325, 43399, 18),
)


def resample_50um_to_25um_physical_fov(img_20x20: np.ndarray, order: int = 3) -> np.ndarray:
    """Resample a 20x20 50 um DIAD slice (`51.316 um/px`, `[-487.5, +487.5] um`) onto the 40x40 25 um grid (`25.0 um/px`).

    Both 25 um (`40` steps) and 50 um (`20` steps) scans span the exact same physical `kb_cs_x` interval
    `[377.72, 2183.28]` DIAD XCT pixels (`975.0 um` center-to-center), so endpoint index `0` maps to `0.0` and
    endpoint index `39` maps to `19.0`.
    """
    arr = np.asarray(img_20x20, dtype=np.float64)
    u = np.linspace(0.0, float(arr.shape[0] - 1), 40, dtype=np.float64)
    yy, xx = np.meshgrid(u, u, indexing="ij")
    warped = map_coordinates(arr, [yy, xx], order=order, mode="nearest")
    return np.clip(warped, 0.0, None).astype(np.float32)


def reconstruct_calibrated_slice(
    nxs_path: Path,
    scan_id: int,
    na_spec: Any,
    zn_spec: Any,
    cap_spec: Any,
    theta_offset_deg: float = 0.0,
) -> Dict[str, Any]:
    """Run continuum-subtracted + q-shifted NNLS unmixing, conjugate COR estimation, and odd/even variance."""
    sino_obj = load_xrdct_sinogram(nxs_path, q_range=(0.85, 3.10), scan_id=scan_id)
    angles = sino_obj.angles_deg + float(theta_offset_deg)
    unmixed = unmix_sinogram_nnls(
        sino_obj.data,
        sino_obj.q_inv_angstrom,
        na_spec,
        zn_spec,
        cap_spec,
        q_bounds=(0.90, 3.05),
        subtract_continuum=True,
        q_shift_inv_a=-0.00125,
        broaden_sigma_bins=1.0,
    )
    bragg_sino = unmixed["na"] + unmixed["zn"]
    cor = estimate_conjugate_cor(bragg_sino, angles)
    zn_rec, zn_s2, zn_frc = estimate_odd_even_noise_floor(unmixed["zn"], angles, center_px=cor.center_px)
    na_rec, na_s2, na_frc = estimate_odd_even_noise_floor(unmixed["na"], angles, center_px=cor.center_px)

    # Deduplicate 0/360 frame for clean sparse-view T4 subsampling
    ang_clean = np.asarray(angles, dtype=np.float64).copy()
    zn_sino = np.asarray(unmixed["zn"], dtype=np.float32).copy()
    na_sino = np.asarray(unmixed["na"], dtype=np.float32).copy()
    if ang_clean.size > 2 and abs(abs(float(ang_clean[-1] - ang_clean[0])) - 360.0) < 0.35:
        zn_sino[0] = 0.5 * (zn_sino[0] + zn_sino[-1])
        na_sino[0] = 0.5 * (na_sino[0] + na_sino[-1])
        zn_sino = zn_sino[:-1]
        na_sino = na_sino[:-1]
        ang_clean = ang_clean[:-1]

    tot_pos = np.clip(zn_rec, 0.0, None) + np.clip(na_rec, 0.0, None)
    pellet_mask = (tot_pos > 0.20 * float(np.max(tot_pos))).astype(np.uint8)
    zn_frac = float(np.sum(np.clip(zn_rec, 0.0, None))) / float(np.sum(tot_pos) + 1e-12)

    return {
        "scan_id": int(scan_id),
        "cor_px": float(cor.center_px),
        "zn": zn_rec.astype(np.float32),
        "na": na_rec.astype(np.float32),
        "zn_sigma2": zn_s2.astype(np.float32),
        "na_sigma2": na_s2.astype(np.float32),
        "pellet_mask": pellet_mask,
        "zn_fraction": round(zn_frac, 4),
        "zn_frc_sb0p5": round(float(zn_frc.cutoff_full_sb_0p5_cyc_per_px), 4),
        "na_frc_sb0p5": round(float(na_frc.cutoff_full_sb_0p5_cyc_per_px), 4),
        "zn_sinogram_unique": zn_sino,
        "na_sinogram_unique": na_sino,
        "angles_unique_deg": ang_clean,
    }


def run_xrdct_classical_suite(
    cache_dir: Path,
    bundles_out_dir: Path,
    results_json_path: Path,
    artifact_png_path: Optional[Path] = None,
    paper_pdf_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute T2 unguided 4-fold LOSO baselines & T4 sparse-view FBP vs SIRT across all cached 25/50 um scans."""
    na_spec = load_ascii_xrd(cache_dir / "13Xhighres_2-Theta_Omega.asc")
    zn_spec = load_ascii_xrd(cache_dir / "Zn13Xhighres_2-Theta_Omega.asc")
    cap_spec = load_ascii_xrd(cache_dir / "empty_cap_2-Theta_Omega.asc")

    recs_25: Dict[int, Dict[str, Any]] = {}
    for sid25, _, _ in PAIRED_25_TO_50:
        p25 = cache_dir / f"k11-{sid25}-diffraction-DiffInt_Mask_Azz_Txt.nxs"
        recs_25[sid25] = reconstruct_calibrated_slice(p25, sid25, na_spec, zn_spec, cap_spec, theta_offset_deg=0.0)

    needed_50 = sorted({s50 for _, s50, _ in PAIRED_25_TO_50} | {43346})
    recs_50: Dict[int, Dict[str, Any]] = {}
    for sid50 in needed_50:
        p50 = cache_dir / f"k11-{sid50}-diffraction-DiffInt_Mask_Azz_Txt_50um.nxs"
        recs_50[sid50] = reconstruct_calibrated_slice(p50, sid50, na_spec, zn_spec, cap_spec, theta_offset_deg=47.5)

    # Save calibrated 25/50 um .npz bundles for OME-Zarr derived release packaging
    bundles_out_dir.mkdir(parents=True, exist_ok=True)
    for sid25, sid50, z_idx in PAIRED_25_TO_50:
        r25 = recs_25[sid25]
        r50 = recs_50[sid50]
        np.savez_compressed(
            bundles_out_dir / f"pair_25um_{sid25}_50um_{sid50}_z{z_idx:02d}.npz",
            scan_25um=np.int32(sid25),
            scan_50um=np.int32(sid50),
            z_index_50um=np.int32(z_idx),
            zn_25um=r25["zn"],
            na_25um=r25["na"],
            zn_sigma2=r25["zn_sigma2"],
            na_sigma2=r25["na_sigma2"],
            pellet_mask_25um=r25["pellet_mask"],
            zn_50um=r50["zn"],
            na_50um=r50["na"],
            zn_50um_sigma2=r50["zn_sigma2"],
            na_50um_sigma2=r50["na_sigma2"],
        )

    # ------------------------------------------------------------------
    # Task T2: Strict 4-fold LOSO unguided classical baselines (Nearest, Bilinear, Bicubic)
    # ------------------------------------------------------------------
    methods_t2 = [("Nearest", 0), ("Bilinear", 1), ("Bicubic", 3)]
    t2_per_method: Dict[str, Any] = {}
    bicubic_preds_for_plot: Dict[int, Dict[str, np.ndarray]] = {}

    for mname, interp_order in methods_t2:
        per_fold: Dict[str, Any] = {}
        for fold_idx, (test_s25, test_s50, test_z) in enumerate(PAIRED_25_TO_50):
            # Fit scalar beam-flux ratio strictly on the 3 TRAINING slice pairs (zero test leakage)
            num_flux = 0.0
            den_flux = 0.0
            for tr_s25, tr_s50, _ in PAIRED_25_TO_50:
                if tr_s25 == test_s25:
                    continue
                up_zn = resample_50um_to_25um_physical_fov(recs_50[tr_s50]["zn"], order=interp_order)
                up_na = resample_50um_to_25um_physical_fov(recs_50[tr_s50]["na"], order=interp_order)
                tot_25 = np.clip(recs_25[tr_s25]["zn"], 0.0, None) + np.clip(recs_25[tr_s25]["na"], 0.0, None)
                tot_50 = up_zn + up_na
                num_flux += float(np.sum(tot_25 * tot_50))
                den_flux += float(np.sum(tot_50 * tot_50))
            alpha_fold = num_flux / max(den_flux, 1e-12)

            pred_zn = resample_50um_to_25um_physical_fov(recs_50[test_s50]["zn"], order=interp_order) * alpha_fold
            pred_na = resample_50um_to_25um_physical_fov(recs_50[test_s50]["na"], order=interp_order) * alpha_fold
            if mname == "Bicubic":
                bicubic_preds_for_plot[test_s25] = {"zn": pred_zn, "na": pred_na, "alpha": np.float32(alpha_fold)}

            ref_b = {
                "zn": np.clip(recs_25[test_s25]["zn"], 0.0, None),
                "na": np.clip(recs_25[test_s25]["na"], 0.0, None),
                "zn_sigma2": recs_25[test_s25]["zn_sigma2"],
                "na_sigma2": recs_25[test_s25]["na_sigma2"],
            }
            ev = evaluate_bundles("t2", {"zn": pred_zn, "na": pred_na}, ref_b, method_name=mname)["metrics"]
            per_fold[str(test_s25)] = {
                "fold_index": fold_idx,
                "matched_50um_scan": test_s50,
                "z_index": test_z,
                "provisional_stage_z": bool(test_s25 == 43324),
                "fitted_flux_scale_alpha": round(alpha_fold, 5),
                "metrics": {k: round(float(v), 5) for k, v in ev.items()},
            }

        keys_agg = (
            "mean_bandlimited_reduced_chi2",
            "mean_bandlimited_rmse",
            "mean_rmse",
            "in_pellet_na_zn_corr_abs_err",
            "reaction_front_pos_err_um",
        )
        confirmed_ids = ["43322", "43323", "43325"]
        all_ids = ["43324", "43322", "43323", "43325"]
        t2_per_method[mname] = {
            "per_slice": per_fold,
            "mean_3_confirmed_slices": {
                k: round(float(np.mean([per_fold[s]["metrics"][k] for s in confirmed_ids])), 4) for k in keys_agg
            },
            "std_3_confirmed_slices": {
                k: round(float(np.std([per_fold[s]["metrics"][k] for s in confirmed_ids])), 4) for k in keys_agg
            },
            "mean_all_4_slices_provisional_43324": {
                k: round(float(np.mean([per_fold[s]["metrics"][k] for s in all_ids])), 4) for k in keys_agg
            },
            "std_all_4_slices_provisional_43324": {
                k: round(float(np.std([per_fold[s]["metrics"][k] for s in all_ids])), 4) for k in keys_agg
            },
        }

    # Sensitivity check on 43324: z=9 (43345) vs z=10 (43346) under Bicubic
    alpha_43324 = float(bicubic_preds_for_plot[43324]["alpha"])
    alt_zn_43346 = resample_50um_to_25um_physical_fov(recs_50[43346]["zn"], order=3) * alpha_43324
    alt_na_43346 = resample_50um_to_25um_physical_fov(recs_50[43346]["na"], order=3) * alpha_43324
    ev_43346 = evaluate_bundles(
        "t2",
        {"zn": alt_zn_43346, "na": alt_na_43346},
        {
            "zn": np.clip(recs_25[43324]["zn"], 0.0, None),
            "na": np.clip(recs_25[43324]["na"], 0.0, None),
            "zn_sigma2": recs_25[43324]["zn_sigma2"],
            "na_sigma2": recs_25[43324]["na_sigma2"],
        },
        method_name="Bicubic_alt_43346",
    )["metrics"]

    # ------------------------------------------------------------------
    # Optional Task T4: Sparse-view sinogram reconstruction (10, 20, 40 views out of 79)
    # ------------------------------------------------------------------
    t4_configs = [("10_views_8x", 8), ("20_views_4x", 4), ("40_views_2x", 2)]
    t4_summary: Dict[str, Any] = {}
    t4_plot_43322: Dict[str, np.ndarray] = {}

    for cfg_name, stride in t4_configs:
        fbp_metrics_list: List[Dict[str, float]] = []
        sirt_metrics_list: List[Dict[str, float]] = []
        per_scan_t4: Dict[str, Any] = {}
        for sid25, _, _ in PAIRED_25_TO_50:
            r25 = recs_25[sid25]
            cor = r25["cor_px"]
            ang_u = r25["angles_unique_deg"]
            sub_ang = ang_u[::stride]
            n_views = int(sub_ang.size)

            scan_entry: Dict[str, Any] = {"num_views": n_views}
            for phase_key, ref_key, s2_key in (
                ("zn", "zn", "zn_sigma2"),
                ("na", "na", "na_sigma2"),
            ):
                sino_sub = r25[f"{phase_key}_sinogram_unique"][::stride]
                ref_img = np.clip(r25[ref_key], 0.0, None)
                ref_s2 = r25[s2_key]

                rec_fbp = np.clip(fbp_sparse_reconstruct_2d(sino_sub, sub_ang, center_px=cor), 0.0, None)
                rec_sirt = sirt_reconstruct_2d(sino_sub, sub_ang, center_px=cor, n_iter=60, relaxation=0.85)

                m_fbp = evaluate_bundles(
                    "t4", {"recon": rec_fbp}, {"recon": ref_img, "sigma2": ref_s2}, f"FBP_{cfg_name}"
                )["metrics"]
                m_sirt = evaluate_bundles(
                    "t4", {"recon": rec_sirt}, {"recon": ref_img, "sigma2": ref_s2}, f"SIRT_{cfg_name}"
                )["metrics"]
                fbp_metrics_list.append(m_fbp)
                sirt_metrics_list.append(m_sirt)
                scan_entry[phase_key] = {
                    "FBP": {k: round(float(v), 4) for k, v in m_fbp.items()},
                    "SIRT": {k: round(float(v), 4) for k, v in m_sirt.items()},
                }
                if sid25 == 43322 and phase_key == "zn" and stride == 4:
                    t4_plot_43322 = {"ref": ref_img, "fbp_20": rec_fbp, "sirt_20": rec_sirt}
            per_scan_t4[str(sid25)] = scan_entry

        keys_t4 = ("reduced_chi2", "rmse", "frc_0p5_cyc_per_px", "frc_full_sb_0p5_cyc_per_px")
        t4_summary[cfg_name] = {
            "angular_stride": stride,
            "approx_views": int(recs_25[43322]["angles_unique_deg"][::stride].size),
            "FBP_mean": {k: round(float(np.mean([m[k] for m in fbp_metrics_list])), 4) for k in keys_t4},
            "FBP_std": {k: round(float(np.std([m[k] for m in fbp_metrics_list])), 4) for k in keys_t4},
            "SIRT_mean": {k: round(float(np.mean([m[k] for m in sirt_metrics_list])), 4) for k in keys_t4},
            "SIRT_std": {k: round(float(np.std([m[k] for m in sirt_metrics_list])), 4) for k in keys_t4},
            "per_scan": per_scan_t4,
        }

    # Render publication diagnostic figure (T2 Bicubic on core-shell 43322 + T4 20-view FBP vs SIRT)
    if artifact_png_path is not None or paper_pdf_path is not None:
        fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), constrained_layout=True)
        r22 = recs_25[43322]
        r50_95 = recs_50[43395]
        bic22 = bicubic_preds_for_plot[43322]["zn"]
        ref22_zn = np.clip(r22["zn"], 0.0, None)
        vmax_zn = float(np.percentile(ref22_zn, 99.5))

        im0 = axes[0, 0].imshow(np.clip(r50_95["zn"], 0.0, None), cmap="viridis", origin="lower")
        axes[0, 0].set_title("a) T2 Input: 50 µm Zn-13X\n(k11-43395, +47.5° aligned)", fontsize=9.5)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

        im1 = axes[0, 1].imshow(bic22, cmap="viridis", vmin=0.0, vmax=vmax_zn, origin="lower")
        axes[0, 1].set_title("b) T2 Unguided Bicubic (2×)\n(LOSO Fold 1 prediction)", fontsize=9.5)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        im2 = axes[0, 2].imshow(ref22_zn, cmap="viridis", vmin=0.0, vmax=vmax_zn, origin="lower")
        axes[0, 2].set_title("c) T2 25 µm Reference\n(k11-43322, FRC≈58.5 µm)", fontsize=9.5)
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

        err_lim = 0.35 * vmax_zn
        im3 = axes[0, 3].imshow(bic22 - ref22_zn, cmap="RdBu_r", vmin=-err_lim, vmax=err_lim, origin="lower")
        axes[0, 3].set_title("d) T2 Residual (Bicubic − Ref)\n(XCT guidance gated on Stage 0C)", fontsize=9.5)
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

        # Row 2: Optional Task T4 Sparse-View Reconstruction (20/79 projections on k11-43322)
        im4 = axes[1, 0].imshow(t4_plot_43322["fbp_20"], cmap="viridis", vmin=0.0, vmax=vmax_zn, origin="lower")
        axes[1, 0].set_title("e) T4 FBP (20/79 views, ~4×)\nStreak artifacts across pores", fontsize=9.5)
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

        im5 = axes[1, 1].imshow(t4_plot_43322["sirt_20"], cmap="viridis", vmin=0.0, vmax=vmax_zn, origin="lower")
        axes[1, 1].set_title("f) T4 Non-Negative SIRT (20/79)\nSuppresses angular streaks", fontsize=9.5)
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

        im6 = axes[1, 2].imshow(
            t4_plot_43322["sirt_20"] - t4_plot_43322["ref"], cmap="RdBu_r", vmin=-err_lim, vmax=err_lim, origin="lower"
        )
        axes[1, 2].set_title("g) T4 Residual (SIRT − 79-view Ref)", fontsize=9.5)
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

        # Panel h: T4 RMSE vs projection count across all 4 scans x 2 phases
        views_x = [t4_summary[k]["approx_views"] for k, _ in t4_configs]
        fbp_y = [t4_summary[k]["FBP_mean"]["rmse"] for k, _ in t4_configs]
        sirt_y = [t4_summary[k]["SIRT_mean"]["rmse"] for k, _ in t4_configs]
        axes[1, 3].plot(views_x, fbp_y, "o--", color="#c0392b", lw=2.0, label="FBP (Hann)")
        axes[1, 3].plot(views_x, sirt_y, "s-", color="#1f77b4", lw=2.0, label="Non-Neg SIRT")
        axes[1, 3].set_xlabel("Subsampled Projections (of 79)")
        axes[1, 3].set_ylabel("Mean RMSE vs 79-view Ref")
        axes[1, 3].set_title("h) T4 Sparse-View Curve\n(4 scans × Na/Zn phases)", fontsize=9.5)
        axes[1, 3].grid(True, alpha=0.3)
        axes[1, 3].legend(frameon=True, fontsize=8.5)

        for r in range(2):
            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])

        if artifact_png_path is not None:
            artifact_png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(artifact_png_path, dpi=180)
        if paper_pdf_path is not None:
            paper_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(paper_pdf_path)
        plt.close(fig)

    payload = {
        "execution_status": "EXECUTED_ON_REAL_32BIT_ZENODO_13329639_SINOGRAMS",
        "t2_unguided_loso_baselines": t2_per_method,
        "t2_43324_stage_z_sensitivity_bicubic": {
            "note": "43324 -> 43345 (z=9) has correlation margin dr=0.0052 over 43346 (z=10); Stage 0A-2 motor check pending on SCARF.",
            "z9_43345_bandlimited_rmse": round(
                float(t2_per_method["Bicubic"]["per_slice"]["43324"]["metrics"]["mean_bandlimited_rmse"]), 4
            ),
            "z10_43346_bandlimited_rmse": round(float(ev_43346["mean_bandlimited_rmse"]), 4),
            "z9_43345_bandlimited_chi2": round(
                float(t2_per_method["Bicubic"]["per_slice"]["43324"]["metrics"]["mean_bandlimited_reduced_chi2"]), 4
            ),
            "z10_43346_bandlimited_chi2": round(float(ev_43346["mean_bandlimited_reduced_chi2"]), 4),
        },
        "t4_sparse_view_baselines": t4_summary,
    }
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run real DIAD XRD-CT T2 & T4 classical baselines.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/zenodo"))
    parser.add_argument("--bundles-out", type=Path, default=Path("data/cache/derived_xrdct"))
    parser.add_argument("--results-json", type=Path, default=Path("outputs/classical_baselines/xrdct_t2_t4_results.json"))
    parser.add_argument("--artifact-png", type=Path, default=None)
    parser.add_argument("--paper-pdf", type=Path, default=Path("paper/figures/fig4_xrdct_t2_t4_baselines.pdf"))
    args = parser.parse_args(argv)

    rep = run_xrdct_classical_suite(
        cache_dir=args.cache_dir,
        bundles_out_dir=args.bundles_out,
        results_json_path=args.results_json,
        artifact_png_path=args.artifact_png,
        paper_pdf_path=args.paper_pdf,
    )
    print(
        json.dumps(
            {
                "t2_bicubic_3confirmed": rep["t2_unguided_loso_baselines"]["Bicubic"]["mean_3_confirmed_slices"],
                "t2_bicubic_all4": rep["t2_unguided_loso_baselines"]["Bicubic"]["mean_all_4_slices_provisional_43324"],
                "t2_43324_sensitivity": rep["t2_43324_stage_z_sensitivity_bicubic"],
                "t4_sparse_view": {
                    k: {"FBP": v["FBP_mean"], "SIRT": v["SIRT_mean"]}
                    for k, v in rep["t4_sparse_view_baselines"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
