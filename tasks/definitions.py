"""Formal Python specifications and submission schema validators for Zn-13X-Bench Tasks T1-T4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import yaml

DEFAULT_TASKS_YAML = Path(__file__).resolve().parents[1] / "configs" / "tasks" / "tasks.yaml"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    name: str
    required_keys: Tuple[str, ...]
    primary_metrics: Tuple[str, ...]
    gated_on_stage0c: bool = False
    optional_extension: bool = False
    classes: Optional[Dict[int, str]] = None
    primary_reference_track: str = "raw_float32"
    secondary_reference_track: str = "cleaned_dering_v1"


def _load_task_specs(yaml_path: Path = DEFAULT_TASKS_YAML) -> Dict[str, TaskSpec]:
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)["tasks"]
    out: Dict[str, TaskSpec] = {}
    for tid, meta in raw.items():
        cls_map = {int(k): str(v) for k, v in meta["classes"].items()} if "classes" in meta else None
        out[tid] = TaskSpec(
            task_id=tid,
            name=str(meta["name"]),
            required_keys=tuple(str(k) for k in meta["required_keys"]),
            primary_metrics=tuple(str(m) for m in meta["primary_metrics"]),
            gated_on_stage0c=bool(meta.get("gated_on_stage0c", False)),
            optional_extension=bool(meta.get("optional_extension", False)),
            classes=cls_map,
            primary_reference_track=str(meta.get("primary_reference_track", "raw_float32")),
            secondary_reference_track=str(meta.get("secondary_reference_track", "cleaned_dering_v1")),
        )
    # Auxiliary diagnostic spec for segmentation helper compatibility (`t4a`)
    out["t4a"] = TaskSpec(
        task_id="t4a",
        name="Auxiliary 3-Phase XCT Segmentation Diagnostic",
        required_keys=("labels",),
        primary_metrics=("macro_dice", "boundary_f1_tol1", "mean_phase_fraction_mae", "d_eff_rel_err"),
        classes={0: "Pore", 1: "Pellet_Solid", 2: "Calibration_Bead"},
    )
    return out


TASKS: Dict[str, TaskSpec] = _load_task_specs()


def get_task_spec(task_id: str, pred_dict: Optional[Mapping[str, np.ndarray]] = None) -> TaskSpec:
    tid = task_id.strip().lower()
    if tid in ("t2_fusion", "chemical_fusion"):
        tid = "t2"
    elif tid in ("t3_registration", "registration"):
        tid = "t3"
    elif tid in ("t4_sparse", "sparse_view"):
        tid = "t4"
    # Smart disambiguation if legacy caller passes "t3" with chemical {"zn", "na"} bundle vs registration bundle:
    elif tid == "t3" and pred_dict is not None and ("zn" in pred_dict and "na" in pred_dict):
        tid = "t2"
    if tid not in TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Valid tasks: {sorted(TASKS.keys())}")
    return TASKS[tid]


def validate_submission_arrays(
    task_id: str,
    pred_dict: Mapping[str, np.ndarray],
    ref_dict: Mapping[str, np.ndarray],
) -> TaskSpec:
    """Validate that a candidate submission contains required arrays, finite values, and exact reference shapes."""
    spec = get_task_spec(task_id, pred_dict=pred_dict)
    for k in spec.required_keys:
        if k not in pred_dict:
            raise ValueError(
                f"Submission for task '{spec.task_id}' missing required key '{k}' (got {list(pred_dict.keys())})"
            )
        if k not in ref_dict:
            raise ValueError(f"Reference bundle for task '{spec.task_id}' missing required key '{k}'")
        pa = np.asarray(pred_dict[k])
        ra = np.asarray(ref_dict[k])
        if pa.shape != ra.shape:
            raise ValueError(f"Shape mismatch on '{k}' for task '{spec.task_id}': pred {pa.shape} vs ref {ra.shape}")
        if not np.all(np.isfinite(pa)):
            raise ValueError(f"Submission array '{k}' contains NaN or Inf values.")
    return spec
