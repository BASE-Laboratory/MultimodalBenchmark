"""Reproducible multi-seed command generator for BASE-Laboratory/OpenLSR-X (T1) and BASE-Laboratory/X-FUSE (T2).

Every generated command matches the exact upstream CLI parsers verified against:
  - `BASE-Laboratory/OpenLSR-X` v0.2.2 (`src/openlsrx/cli.py` & `src/openlsrx/models/srgan.py`):
      * `openlsrx fit --config <yaml> --seed_everything <seed> --model.backbone {srgan,rrdb,swinir} ...`
      * `openlsrx predict-volume --checkpoint <ckpt> --input <lr_tif> --output <pred_tif> --fusion confidence_fourier`
      * `python evaluate.py --task t1 --prediction <pred_tif> --reference <ref_tif> --threshold-sensitivity ...`
  - `BASE-Laboratory/X-FUSE` v1.0.0 (`x_fuse/cli.py`):
      * `xfuse fuse --hr <hr_tif> --guide <zn_tif> <na_tif> --output <out.zarr> --conformal-alpha 0.10 ...`
      * `xfuse fuse-3d --hr-volume <xct_h5> --guide-slice <z>:<path> ... --output <out_3d.zarr> --conformal-alpha 0.10`
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

VALID_OPENLSRX_BACKBONES = {"srgan", "rrdb", "swinir"}
VALID_XFUSE_METHODS = {"gating", "learned_gating", "spatial_mlp", "attention"}
VALID_XFUSE_UPSAMPLERS = {"jbu", "cross_attn", "bilinear", "bicubic", "nearest", "nearest-exact", "area"}


def build_openlsrx_commands(
    config_path: Path | str,
    out_root: Path | str,
    include_exploratory_8x: bool = False,
    with_mechanics: bool = False,
) -> list[str]:
    """Generate real `openlsrx fit` -> `openlsrx predict-volume` -> `evaluate.py` pipelines for Task T1."""
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seeds: list[int] = [int(s) for s in cfg.get("seeds", [42, 137, 2026])]
    tracks: list[dict[str, Any]] = list(cfg["tracks"])
    regimes: list[dict[str, Any]] = list(cfg["training_regimes"])
    models: list[dict[str, Any]] = list(cfg["models"])
    data_dir = str(cfg.get("prepared_tiffs_dir", "data/benchmarks/zn13x_float32"))
    mech_flag = " --with-mechanics" if with_mechanics else ""

    cmds: list[str] = []
    for trk in tracks:
        if trk.get("exploratory", False) and not include_exploratory_8x:
            continue
        tname = str(trk["name"])
        base_cfg = str(trk["openlsrx_base_config"])
        hr_vox = float(trk["ruby_calibrated_hr_voxel_um"])
        eval_lr_tif = f"{data_dir}/{tname}_real_paired_test_lr.tif"
        eval_hr_tif = f"{data_dir}/{tname}_real_paired_test_hr.tif"

        for reg in regimes:
            rname = str(reg["name"])
            train_lr_tif = f"{data_dir}/{tname}_{rname}_train_lr.tif"
            train_hr_tif = f"{data_dir}/{tname}_{rname}_train_hr.tif"

            for mdl in models:
                mname = str(mdl["name"])
                bb = str(mdl["backbone"])
                if bb not in VALID_OPENLSRX_BACKBONES:
                    raise ValueError(
                        f"Invalid OpenLSR-X backbone '{bb}' in {mname}; must be one of {sorted(VALID_OPENLSRX_BACKBONES)}"
                    )
                ctx = int(mdl.get("context_slices", 3))
                w_cons = float(mdl.get("conservation_loss_weight", 0.0))
                w_topo = float(mdl.get("topology_loss_weight", 0.0))
                w_ffl = float(mdl.get("ffl_loss_weight", 0.0))
                use_gan = str(bool(mdl.get("use_adversarial_loss", False))).lower()

                for seed in seeds:
                    run_dir = Path(out_root) / f"t1_{tname}_{rname}_{mname}_seed{seed}"
                    ckpt = f"{run_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
                    pred_tif = f"{run_dir}/pred_test.tiff"
                    std_tif = f"{run_dir}/pred_test_sigma_ortho.tiff"
                    score_json = f"{run_dir}/zn13x_bench_t1_score.json"

                    fit_cmd = (
                        f"openlsrx fit --config {base_cfg} "
                        f"--seed_everything {seed} "
                        f"--model.backbone {bb} "
                        f"--model.context_slices {ctx} "
                        f"--model.use_adversarial_loss {use_gan} "
                        f"--model.conservation_loss_weight {w_cons} "
                        f"--model.topology_loss_weight {w_topo} "
                        f"--model.ffl_loss_weight {w_ffl} "
                        f"--data.init_args.low_path {train_lr_tif} "
                        f"--data.init_args.high_path {train_hr_tif} "
                        f"--data.init_args.norm_mode none "
                        f"--data.init_args.cross_scan_calibration none "
                        f"--data.init_args.seed {seed} "
                        f"--trainer.default_root_dir {run_dir}"
                    )
                    pred_cmd = (
                        f"openlsrx predict-volume "
                        f"--checkpoint {ckpt} "
                        f"--input {eval_lr_tif} "
                        f"--output {pred_tif} "
                        f"--uncertainty-output {std_tif} "
                        f"--fusion confidence_fourier"
                    )
                    eval_cmd = (
                        f"python evaluate.py --task t1 "
                        f"--prediction {pred_tif} "
                        f"--reference {eval_hr_tif} "
                        f"--voxel-size-um {hr_vox} "
                        f"--threshold-sensitivity{mech_flag} "
                        f"--method {mname}_{rname} "
                        f"--out-json {score_json}"
                    )
                    cmds.append(f"{fit_cmd} && {pred_cmd} && {eval_cmd}")
    return cmds


def build_xfuse_commands(
    config_path: Path | str,
    out_root: Path | str,
    conformal_alpha: float = 0.10,
) -> list[str]:
    """Generate real `xfuse fuse` / `xfuse fuse-3d` commands for Task T2 4-fold LOSO chemical fusion."""
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seeds: list[int] = [int(s) for s in cfg.get("seeds", [42, 137, 2026])]
    variants: list[dict[str, Any]] = list(cfg["variants"])
    folds: list[dict[str, Any]] = list(cfg["loso_folds"])
    data_dir = str(cfg.get("prepared_slices_dir", "data/benchmarks/zn13x_t2_diad"))
    epochs_2d = int(cfg.get("epochs_2d", 25))
    epochs_3d = int(cfg.get("epochs_3d", 12))
    device = str(cfg.get("device", "cuda"))
    conf_alpha = float(cfg.get("conformal_alpha", conformal_alpha))

    cmds: list[str] = []
    for var in variants:
        vname = str(var["name"])
        mode = str(var.get("cli_subcommand", "fuse"))
        backbone = str(var.get("backbone", "dinov2_vits14"))

        for fold in folds:
            fidx = int(fold["fold_index"])
            holdout = int(fold["test_25um_scan"])
            matched_50 = int(fold["matched_50um_scan"])
            z_idx = int(fold["z_index_50um"])

            for seed in seeds:
                run_dir = Path(out_root) / f"t2_{vname}_fold{fidx}_holdout{holdout}_seed{seed}"
                score_json = f"{run_dir}/zn13x_bench_t2_score.json"
                ref_bundle = f"{data_dir}/ref_25um_{holdout}.ome.zarr"
                if mode == "fuse":
                    fmethod = str(var.get("fusion_method", "spatial_mlp"))
                    upsamp = str(var.get("upsampler", "cross_attn"))
                    if fmethod not in VALID_XFUSE_METHODS:
                        raise ValueError(f"Invalid xfuse --fusion-method '{fmethod}'")
                    if upsamp not in VALID_XFUSE_UPSAMPLERS:
                        raise ValueError(f"Invalid xfuse --upsampler '{upsamp}'")
                    out_zarr = f"{run_dir}/pred_2d_{holdout}.ome.zarr"
                    fuse_cmd = (
                        f"PYTHONHASHSEED={seed} xfuse fuse "
                        f"--hr {data_dir}/xct_slab_z{z_idx:02d}_{holdout}.tif "
                        f"--guide {data_dir}/zn_50um_{matched_50}.tif {data_dir}/na_50um_{matched_50}.tif "
                        f"--output {out_zarr} "
                        f"--backbone {backbone} "
                        f"--fusion-method {fmethod} "
                        f"--loss compositional_kl "
                        f"--upsampler {upsamp} "
                        f"--pore-mask auto_otsu_fringe_guarded "
                        f"--conformal-alpha {conf_alpha:.2f} "
                        f"--pixel-size-um 0.54 "
                        f"--epochs {epochs_2d} "
                        f"--device {device}"
                    )
                    eval_cmd = (
                        f"python evaluate.py --task t2 "
                        f"--prediction {out_zarr} "
                        f"--reference {ref_bundle} "
                        f"--method {vname}_fold{fidx} "
                        f"--out-json {score_json}"
                    )
                    cmds.append(f"{fuse_cmd} && {eval_cmd}")
                elif mode == "fuse-3d":
                    guide_specs = " ".join(
                        f"{int(tr['z_index_50um'])}:{data_dir}/stack_zn_na_25um_{int(tr['scan_25um'])}.tif"
                        for tr in fold["train_slices"]
                    )
                    out_zarr = f"{run_dir}/pred_3d_fold{fidx}.ome.zarr"
                    fuse_cmd = (
                        f"PYTHONHASHSEED={seed} xfuse fuse-3d "
                        f"--hr-volume {data_dir}/diad_xct_subvolume.h5 "
                        f"--guide-slice {guide_specs} "
                        f"--output {out_zarr} "
                        f"--backbone {backbone} "
                        f"--loss compositional_kl "
                        f"--pore-mask auto_otsu_fringe_guarded "
                        f"--conformal-alpha {conf_alpha:.2f} "
                        f"--epochs {epochs_3d} "
                        f"--device {device}"
                    )
                    eval_cmd = (
                        f"python evaluate.py --task t2 "
                        f"--prediction {out_zarr} "
                        f"--reference {ref_bundle} "
                        f"--method {vname}_fold{fidx} "
                        f"--out-json {score_json}"
                    )
                    cmds.append(f"{fuse_cmd} && {eval_cmd}")
                else:
                    raise ValueError(f"Unsupported X-FUSE CLI subcommand '{mode}' (must be 'fuse' or 'fuse-3d')")
    return cmds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate verified OpenLSR-X (T1) and X-FUSE (T2) CLI commands.")
    parser.add_argument("--suite", choices=["openlsrx", "xfuse"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/learned_baselines"))
    parser.add_argument("--include-exploratory-8x", action="store_true")
    parser.add_argument("--with-mechanics", action="store_true")
    args = parser.parse_args(argv)

    cmds = (
        build_openlsrx_commands(
            args.config,
            args.out_root,
            include_exploratory_8x=args.include_exploratory_8x,
            with_mechanics=args.with_mechanics,
        )
        if args.suite == "openlsrx"
        else build_xfuse_commands(args.config, args.out_root)
    )
    payload: dict[str, Any] = {"suite": args.suite, "num_runs": len(cmds), "commands": cmds}
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
