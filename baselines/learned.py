"""Reproducible 3-seed configuration & adapter CLI for BASE-Laboratory learned baselines (OpenLSR-X & X-FUSE)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


def build_openlsrx_commands(config_path: Path | str, out_root: Path | str) -> List[str]:
    """Generate deterministic 3-seed training/inference commands for BASE-Laboratory/OpenLSR-X."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seeds: List[int] = cfg.get("seeds", [42, 137, 2025])
    backbones: List[str] = cfg.get("backbones", ["edsr3d", "rcan3d", "swinir3d"])
    track = cfg.get("track", "2x")
    splits_json = cfg.get("splits_file", "splits/benchmark_splits_v1.json")

    cmds: List[str] = []
    for bb in backbones:
        for seed in seeds:
            run_dir = Path(out_root) / f"openlsrx_{track}_{bb}_seed{seed}"
            cmds.append(
                f"python -m openlsrx.cli train_and_predict "
                f"--dataset zn13x_benchmark --splits {splits_json} "
                f"--track {track} --backbone {bb} --seed {seed} "
                f"--out-dir {run_dir}"
            )
    return cmds


def build_xfuse_commands(config_path: Path | str, out_root: Path | str) -> List[str]:
    """Generate deterministic 3-seed self-supervised fusion commands for BASE-Laboratory/X-FUSE."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seeds: List[int] = cfg.get("seeds", [42, 137, 2025])
    splits_json = cfg.get("splits_file", "splits/benchmark_splits_v1.json")
    flip_x = bool(cfg.get("diad_astra_flip_x", True))

    cmds: List[str] = []
    for seed in seeds:
        run_dir = Path(out_root) / f"xfuse_t3_seed{seed}"
        cmds.append(
            f"python -m x_fuse.cli fit_correlative "
            f"--splits {splits_json} --aggregation all_sum "
            f"--flip-x {str(flip_x).lower()} --conformal-alpha 0.10 "
            f"--seed {seed} --out-dir {run_dir}"
        )
    return cmds


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or launch 3-seed OpenLSR-X / X-FUSE baseline runs.")
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
