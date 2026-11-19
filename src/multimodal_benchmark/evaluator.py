"""Unified benchmark submission evaluator and leaderboard formatter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import h5py
import numpy as np

from .metrics import (
    compute_image_sr_metrics,
    compute_segmentation_metrics,
    compute_t3_fusion_metrics,
    compute_transport_discrepancy,
)
from tasks.definitions import get_task_spec, validate_submission_arrays


def load_bundle(path: Path | str) -> Dict[str, np.ndarray]:
    """Load a dictionary of named arrays from either a .npz or .h5/.hdf5/.nxs submission file."""
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as data:
            return {k: np.asarray(data[k]) for k in data.files}
    out: Dict[str, np.ndarray] = {}
    with h5py.File(p, "r") as h5f:
        def _collect(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                out[name] = np.asarray(obj[:])
        h5f.visititems(_collect)
    return out


def bootstrap_ci_95(values: Sequence[float], n_boot: int = 500, seed: int = 42) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    if arr.size == 1 or n_boot <= 0:
        val = float(arr[0])
        return {"mean": val, "ci95_low": val, "ci95_high": val}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = np.mean(arr[idx], axis=1)
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def evaluate_bundles(
    task_id: str,
    pred_bundle: Mapping[str, np.ndarray],
    ref_bundle: Mapping[str, np.ndarray],
    method_name: str = "candidate",
    voxel_size_um: Optional[float] = None,
    with_transport: bool = True,
    bootstrap_samples: int = 200,
) -> Dict[str, Any]:
    """Validate and score a candidate prediction bundle against a ground-truth reference bundle."""
    spec = get_task_spec(task_id)
    validate_submission_arrays(spec.task_id, pred_bundle, ref_bundle)

    mask = ref_bundle.get("mask")
    metrics: Dict[str, float] = {}
    ci95: Dict[str, Dict[str, float]] = {}

    if spec.task_id in ("t1", "t2"):
        vox = voxel_size_um if voxel_size_um is not None else (0.8125 if spec.task_id == "t1" else 0.54)
        pred = pred_bundle["pred"]
        ref = ref_bundle["pred"]
        metrics.update(compute_image_sr_metrics(pred, ref, mask=mask, voxel_size_um=vox))
        if with_transport and ref.ndim == 3:
            metrics.update(
                compute_transport_discrepancy(
                    pred,
                    ref,
                    inputs_are_grayscale=True,
                    voxel_size_um=vox,
                    mask=mask,
                )
            )
        if bootstrap_samples > 0 and ref.ndim == 3 and ref.shape[0] >= 4:
            slice_psnr = [
                compute_image_sr_metrics(pred[z], ref[z], mask=mask[z] if mask is not None else None)["psnr_db"]
                for z in range(ref.shape[0])
            ]
            ci95["psnr_db"] = bootstrap_ci_95(slice_psnr, n_boot=bootstrap_samples)

    elif spec.task_id == "t3":
        metrics.update(
            compute_t3_fusion_metrics(
                pred_zn_25um=pred_bundle["zn"],
                pred_na_25um=pred_bundle["na"],
                ref_zn_25um=ref_bundle["zn"],
                ref_na_25um=ref_bundle["na"],
                ref_zn_sigma2=ref_bundle.get("zn_sigma2"),
                ref_na_sigma2=ref_bundle.get("na_sigma2"),
                hr_pred_zn=pred_bundle.get("hr_zn"),
                hr_pred_na=pred_bundle.get("hr_na"),
                hr_pore_mask=ref_bundle.get("hr_pore_mask"),
                hr_solid_mask=ref_bundle.get("hr_solid_mask"),
            )
        )
        if bootstrap_samples > 0 and ref_bundle["zn"].ndim == 3 and ref_bundle["zn"].shape[0] >= 2:
            per_slice_rmse = [
                compute_t3_fusion_metrics(
                    pred_bundle["zn"][i],
                    pred_bundle["na"][i],
                    ref_bundle["zn"][i],
                    ref_bundle["na"][i],
                )["mean_rmse"]
                for i in range(ref_bundle["zn"].shape[0])
            ]
            ci95["mean_rmse"] = bootstrap_ci_95(per_slice_rmse, n_boot=bootstrap_samples)

    elif spec.task_id in ("t4a", "t4b"):
        vox = voxel_size_um if voxel_size_um is not None else 0.54
        pred_lbl = pred_bundle["labels"]
        ref_lbl = ref_bundle["labels"]
        metrics.update(
            compute_segmentation_metrics(
                pred_lbl,
                ref_lbl,
                class_names=spec.classes,
                mask=mask,
            )
        )
        if with_transport and ref_lbl.ndim == 3:
            metrics.update(
                compute_transport_discrepancy(
                    (pred_lbl == 0).astype(np.uint8),
                    (ref_lbl == 0).astype(np.uint8),
                    inputs_are_grayscale=False,
                    voxel_size_um=vox,
                )
            )

    return {
        "task": spec.task_id,
        "task_name": spec.name,
        "method": method_name,
        "primary_metrics": list(spec.primary_metrics),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "bootstrap_ci95": ci95,
    }


def format_markdown_leaderboard_row(result: Mapping[str, Any]) -> str:
    """Format an evaluation result dictionary as a two-line Markdown header + leaderboard row."""
    p_keys: List[str] = [k for k in result["primary_metrics"] if k in result["metrics"]]
    if not p_keys:
        p_keys = list(result["metrics"].keys())[:5]
    headers = ["Method", "Task"] + p_keys
    sep = ["---"] * len(headers)
    vals = [str(result["method"]), str(result["task"]).upper()]
    for k in p_keys:
        v = result["metrics"][k]
        vals.append(f"{v:.4f}" if np.isfinite(v) else str(v))
    return (
        "| " + " | ".join(headers) + " |\n"
        "| " + " | ".join(sep) + " |\n"
        "| " + " | ".join(vals) + " |"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score a MultimodalBenchmark submission.")
    parser.add_argument("--task", required=True, help="Task ID: t1, t2, t3, t4a, or t4b")
    parser.add_argument("--prediction", type=Path, required=True, help="Path to prediction .npz or .h5")
    parser.add_argument("--reference", type=Path, required=True, help="Path to ground-truth reference .npz or .h5")
    parser.add_argument("--method", default="candidate", help="Method name for leaderboard row")
    parser.add_argument("--voxel-size-um", type=float, default=None, help="Physical voxel size in um")
    parser.add_argument("--no-transport", action="store_true", help="Skip OpenImpala PDE transport metrics")
    parser.add_argument("--bootstrap-samples", type=int, default=200, help="Bootstrap resamples for 95%% CIs")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown leaderboard table")
    args = parser.parse_args(argv)

    pred_b = load_bundle(args.prediction)
    ref_b = load_bundle(args.reference)
    result = evaluate_bundles(
        task_id=args.task,
        pred_bundle=pred_b,
        ref_bundle=ref_b,
        method_name=args.method,
        voxel_size_um=args.voxel_size_um,
        with_transport=not args.no_transport,
        bootstrap_samples=args.bootstrap_samples,
    )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    if args.markdown:
        print(format_markdown_leaderboard_row(result))
    else:
        print(json.dumps(result, indent=2))
    return 0
