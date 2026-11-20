"""Reproducible 3-seed command generator for Zn-13X-Bench learned baselines (T1 Synthetic-vs-Real & T2 LOSO Fusion)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


def build_openlsrx_commands(config_path: Path | str, out_root: Path | str) -> List[str]:
    """Generate 3-seed Synthetic-vs-Real training & real-pair evaluation commands for Task T1."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seeds: List[int] = cfg.get("seeds", [42, 137, 2026])
    backbones: List[str] = cfg.get("backbones", ["unet3d", "swinir3d", "openlsrx"])
    tracks: List[str] = cfg.get("tracks", [cfg.get("track", "2p026x")])
    regimes: List[str] = cfg.get("training_regimes", ["synthetic_clean", "synthetic_noisy", "real_paired"])
    splits_json = cfg.get("splits_file", "splits/benchmark_splits_v1.json")
    tfm_json = cfg.get("benchmark_transforms_file", "configs/i13_benchmark_transforms.json")
    ref_track = cfg.get("reference_track", "raw_float32")

    cmds: List[str] = []
    for trk in tracks:
        for reg in regimes:
            for bb in backbones:
                for seed in seeds:
                    run_dir = Path(out_root) / f"t1_{trk}_{reg}_{bb}_seed{seed}"
                    cmds.append(
                        f"python -m openlsrx.cli train_and_predict "
                        f"--dataset zn13x_bench --splits {splits_json} "
                        f"--benchmark-transforms {tfm_json} --reference-track {ref_track} "
                        f"--track {trk} --train-regime {reg} --eval-regime real_paired "
                        f"--backbone {bb} --seed {seed} --out-dir {run_dir}"
                    )
    return cmds


def build_xfuse_commands(config_path: Path | str, out_root: Path | str) -> List[str]:
    """Generate 4-fold Leave-One-Slice-Out (LOSO) 3-seed commands for Task T2 chemical fusion."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seeds: List[int] = cfg.get("seeds", [42, 137, 2026])
    models: List[str] = cfg.get("models", ["xfuse"])
    loso_slices: List[int] = cfg.get("loso_25um_slices", cfg.get("held_out_25um_test_scans", [43324, 43322, 43323, 43325]))
    splits_json = cfg.get("splits_file", "splits/benchmark_splits_v1.json")
    cutoff = float(cfg.get("evaluation_cutoff_cyc_per_px", 0.43))

    cmds: List[str] = []
    for model_name in models:
        for fold_idx, holdout_scan in enumerate(loso_slices):
            for seed in seeds:
                run_dir = Path(out_root) / f"t2_{model_name}_fold{fold_idx}_holdout{holdout_scan}_seed{seed}"
                cmds.append(
                    f"python -m x_fuse.cli fit_correlative "
                    f"--splits {splits_json} --loso-fold {fold_idx} --holdout-scan {holdout_scan} "
                    f"--model {model_name} --frc-cutoff-cyc-px {cutoff:.2f} "
                    f"--seed {seed} --out-dir {run_dir}"
                )
    return cmds


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Zn-13X-Bench T1/T2 learned baseline experiment commands.")
    parser.add_argument("--suite", choices=["openlsrx", "xfuse"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/learned_baselines"))
    args = parser.parse_args(argv)

    cmds = (
        build_openlsrx_commands(args.config, args.out_root)
        if args.suite == "openlsrx"
        else build_xfuse_commands(args.config, args.out_root)
    )
    payload: Dict[str, Any] = {"suite": args.suite, "num_runs": len(cmds), "commands": cmds}
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
