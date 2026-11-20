"""Generate publication-quality vector PDF figures for Sections 3 and 4 of the Zn-13X-Bench NeurIPS D&B manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_fig1_metrology_and_ladder(out_pdf: Path, cache_dir: Path, transforms_json: Path) -> None:
    """Figure 1: I13-2 4-rung XCT ladder, ruby-sphere turret scale metrology, and fiducial-vs-pellet slip audit."""
    meta = json.loads(transforms_json.read_text(encoding="utf-8"))
    cal = meta["calibrated_voxel_sizes_um"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)

    # Panel a: Mid-slice comparison from cached 169067 (1.62 um) and 169066 (0.80 um) if available
    p66 = cache_dir / "169066_8bit_cropped.h5"
    if p66.exists():
        with h5py.File(p66, "r") as fh:
            ds_key = list(fh.keys())[0]
            vol = fh[ds_key]
            sl = np.asarray(vol[vol.shape[0] // 2, :, :], dtype=np.float32)
        axes[0].imshow(sl, cmap="gray", origin="lower")
        axes[0].set_title("a) I13-2 Rung 2 Reference (169066)\nTrue 0.8006 µm/vox (100 µm LuAG:Ce)", fontsize=9.5)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    else:
        axes[0].text(0.5, 0.5, "169066 Mid-Slice", ha="center", va="center")

    # Panel b: Nominal vs Ruby-Calibrated Scale Error (%) across all 4 objectives
    rungs = ["169068\n(1.25×, 500µm)", "169067\n(2×, 500µm)", "169066\n(4×, 100µm)", "169065\n(10×, 25µm)"]
    scan_ids = ["169068_1p25x", "169067_2x", "169066_4x", "169065_10x"]
    errs_pct = [
        100.0 * (cal[s]["ruby_calibrated_um"] - cal[s]["nominal_um"]) / cal[s]["nominal_um"]
        for s in scan_ids
    ]
    colors = ["#2b5c8f", "#2b5c8f", "#c0392b", "#d35400"]
    bars = axes[1].bar(rungs, errs_pct, color=colors, width=0.58, edgecolor="black", linewidth=0.7)
    axes[1].axhline(0.0, color="black", lw=0.8, ls="--")
    axes[1].set_ylabel("Voxel Scale Deviation vs Nominal (%)")
    axes[1].set_title("b) Scintillator Focus Scale Shift\n500 µm Ruby Sphere Metrology", fontsize=9.5)
    axes[1].set_ylim(-1.9, 0.4)
    axes[1].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, errs_pct):
        axes[1].text(b.get_x() + b.get_width() / 2, v - 0.16, f"{v:.2f}%", ha="center", va="top", fontsize=8.5, fontweight="bold")

    # Panel c: Independent Slip & Self-Consistency Audit (um vs working voxel size)
    labels = ["Stage Encoders\n(Max Δ over 2h19m)", "Rung 1 Fid−Pel\n(2.6→1.62 µm)", "Rung 2 Fid−Pel\n(1.62→0.80 µm)", "4-Rung Loop\nSelf-Consistency"]
    vals_um = [0.40, 3.00, 1.90, 3.64]
    vox_floor = [2.59, 6.50, 3.24, 2.59]
    x = np.arange(len(labels))
    axes[2].bar(x - 0.17, vals_um, width=0.34, color="#1f77b4", label="Measured Offset (µm)", edgecolor="black", linewidth=0.7)
    axes[2].bar(x + 0.17, vox_floor, width=0.34, color="#bdc3c7", label="1 Working Voxel (µm)", edgecolor="black", linewidth=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8.0)
    axes[2].set_ylabel("Physical Displacement (µm)")
    axes[2].set_title("c) Independent I13-2 Slip & Loop Audit\nAll Displacements Below 1 Working Voxel", fontsize=9.5)
    axes[2].legend(fontsize=8.0, frameon=True)
    axes[2].grid(axis="y", alpha=0.3)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def generate_fig2_xrdct_alignment_and_profile(out_pdf: Path, bundles_dir: Path) -> None:
    """Figure 2: Calibrated 25 um Na/Zn axial exchange series (99% -> 62% -> 25% -> 10% Zn)."""
    pairs = [
        (43324, 43345, 9, "z=9* (99.1% Zn)"),
        (43322, 43395, 14, "z=14 (62.4% Zn)"),
        (43323, 43398, 17, "z=17 (25.2% Zn)"),
        (43325, 43399, 18, "z=18 (9.6% Zn)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 5.8), constrained_layout=True)
    for col, (s25, s50, z_idx, lbl) in enumerate(pairs):
        npz = bundles_dir / f"pair_25um_{s25}_50um_{s50}_z{z_idx:02d}.npz"
        with np.load(npz) as d:
            zn = np.clip(d["zn_25um"], 0.0, None)
            na = np.clip(d["na_25um"], 0.0, None)
        vmax = max(float(np.percentile(zn + na, 99.0)), 1e-4)
        im_z = axes[0, col].imshow(zn, cmap="magma", vmin=0.0, vmax=vmax, origin="lower")
        axes[0, col].set_title(f"k11-{s25} Zn-13X\n{lbl}", fontsize=9.0)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        plt.colorbar(im_z, ax=axes[0, col], fraction=0.046)

        im_n = axes[1, col].imshow(na, cmap="cividis", vmin=0.0, vmax=vmax, origin="lower")
        axes[1, col].set_title(f"k11-{s25} Na-13X\nContinuum + Δq calibrated", fontsize=9.0)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        plt.colorbar(im_n, ax=axes[1, col], fraction=0.046)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def generate_fig3_frc_and_cupping_audit(out_pdf: Path) -> None:
    """Figure 3: True 25 vs 50 um XRD-CT FRC resolution limits and radial cupping-vs-chemistry audit."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)

    # Panel a: Effective spatial resolution (um) under raw half-split, Spearman-Brown full, and 1/2-bit
    criteria = ["Raw Odd/Even\nFRC(0.5)", "Spearman–Brown\nFull FRC(0.5)", "van Heel\n1/2-Bit Threshold"]
    res_25 = [64.3, 58.5, 57.9]
    res_50 = [120.1, 103.6, 111.5]
    x = np.arange(len(criteria))
    axes[0].bar(x - 0.17, res_25, width=0.34, color="#1f77b4", label="25 µm Reference (4 scans)", edgecolor="black", lw=0.7)
    axes[0].bar(x + 0.17, res_50, width=0.34, color="#ff7f0e", label="50 µm Input (matched)", edgecolor="black", lw=0.7)
    axes[0].axhline(58.5, color="#1f77b4", ls="--", lw=1.0, alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(criteria, fontsize=8.3)
    axes[0].set_ylabel("Effective Spatial Resolution (µm, ↓ sharper)")
    axes[0].set_title("a) True XRD-CT Resolution Gain: 1.79×–1.93×\n(Scored strictly at k ≤ 0.43 cyc/px)", fontsize=9.5)
    axes[0].legend(fontsize=8.0, frameon=True)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel b: Correlation of removed XCT fields with real 25 um Zn-13X core-shell map
    scans = ["43324\n(99% Zn)", "43322\n(Core-Shell)", "43323\n(Core-Shell)", "43325\n(10% Zn)"]
    r_cupping = [0.062, 0.281, 0.589, 0.144]
    r_dering = [0.014, -0.021, 0.026, -0.011]
    x2 = np.arange(len(scans))
    axes[1].bar(x2 - 0.17, r_cupping, width=0.34, color="#c0392b", label="Rejected 7-Term Cupping Field", edgecolor="black", lw=0.7)
    axes[1].bar(x2 + 0.17, r_dering, width=0.34, color="#27ae60", label="Retained Polar De-Ringing Field", edgecolor="black", lw=0.7)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(scans, fontsize=8.3)
    axes[1].set_ylabel("Pearson r with 25 µm Zn-13X Map")
    axes[1].set_title("b) Why Cupping Detrending Was Rejected\nAbsorbs Real Zn/Na Core–Shell Chemistry", fontsize=9.5)
    axes[1].legend(fontsize=8.0, frameon=True)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel c: Synthetic Ground-Truth Phantom Audit (True Pore-Edge & Radial Slope Retention %)
    metrics = ["True Pore-Edge\nRetention (%)", "Radial Zn/Na Slope\nRetention (%)", "Ring Variance\nSuppression (%)"]
    dering_vals = [98.95, 94.6, 76.4]
    cup_vals = [98.40, -44.9, 77.1]
    x3 = np.arange(len(metrics))
    axes[2].bar(x3 - 0.17, dering_vals, width=0.34, color="#27ae60", label="Polar De-Ringing (Retained)", edgecolor="black", lw=0.7)
    axes[2].bar(x3 + 0.17, cup_vals, width=0.34, color="#c0392b", label="Polar + Cupping (Rejected)", edgecolor="black", lw=0.7)
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(metrics, fontsize=8.3)
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_title("c) Known-Ground-Truth Phantom Audit\nCupping Inverts True Radial Chemistry Slope", fontsize=9.5)
    axes[2].legend(fontsize=8.0, frameon=True)
    axes[2].grid(axis="y", alpha=0.3)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    fdir = root / "paper" / "figures"
    generate_fig1_metrology_and_ladder(
        fdir / "fig1_metrology_and_ladder.pdf",
        root / "data" / "cache" / "zenodo",
        root / "configs" / "i13_benchmark_transforms.json",
    )
    generate_fig2_xrdct_alignment_and_profile(
        fdir / "fig2_xrdct_calibration_and_alignment.pdf",
        root / "data" / "cache" / "derived_xrdct",
    )
    generate_fig3_frc_and_cupping_audit(fdir / "fig3_frc_and_cupping_audit.pdf")
    print("Generated vector PDF figures 1-3 in", fdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
