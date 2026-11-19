"""Formal Python specifications and submission schema validators for Tasks T1-T4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

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
    classes: Optional[Dict[int, str]] = None


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
            classes=cls_map,
        )
    return out


TASKS: Dict[str, TaskSpec] = _load_task_specs()


def get_task_spec(task_id: str) -> TaskSpec:
    tid = task_id.strip().lower()
    if tid == "t4":
        tid = "t4b"
    if tid not in TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Valid tasks: {sorted(TASKS.keys())}")
    return TASKS[tid]


def validate_submission_arrays(
    task_id: str,
    pred_dict: Mapping[str, np.ndarray],
    ref_dict: Mapping[str, np.ndarray],
) -> None:
    """Validate that a candidate submission contains required arrays, finite values, and exact reference shapes."""
    spec = get_task_spec(task_id)
    for k in spec.required_keys:
        if k not in pred_dict:
            raise ValueError(f"Submission for task '{spec.task_id}' missing required key '{k}' (got {list(pred_dict.keys())})")
        if k not in ref_dict:
            raise ValueError(f"Reference bundle for task '{spec.task_id}' missing required key '{k}'")
        pa = np.asarray(pred_dict[k])
        ra = np.asarray(ref_dict[k])
        if pa.shape != ra.shape:
            raise ValueError(f"Shape mismatch on '{k}' for task '{spec.task_id}': pred {pa.shape} vs ref {ra.shape}")
        if not np.all(np.isfinite(pa)):
            raise ValueError(f"Submission array '{k}' contains NaN or Inf values.")
