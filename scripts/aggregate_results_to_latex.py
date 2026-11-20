"""Aggregate multi-seed / multi-fold `evaluate.py` JSON outputs into NeurIPS `booktabs` LaTeX tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def aggregate_score_jsons(score_files: Sequence[Path]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Group `evaluate.py` JSON outputs by `(task, method)` and compute `mean` and `std` across seeds/folds."""
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    meta_task: Dict[str, str] = {}
    for p in score_files:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        task = str(doc.get("task", "unknown")).upper()
        method = str(doc.get("method", p.stem))
        key = f"{task}::{method}"
        meta_task[key] = task
        grouped[key].append({k: float(v) for k, v in doc.get("metrics", {}).items() if isinstance(v, (int, float))})

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, runs in sorted(grouped.items()):
        metric_names = sorted({mk for r in runs for mk in r.keys()})
        stats: Dict[str, Dict[str, float]] = {}
        for mk in metric_names:
            vals = [r[mk] for r in runs if mk in r and np.isfinite(r[mk])]
            if vals:
                stats[mk] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                    "n": float(len(vals)),
                }
        summary[key] = stats
    return summary


def format_booktabs_table(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    metrics_cols: Sequence[str],
    caption: str = "Zn-13X-Bench Multi-Seed Leaderboard Results",
    label: str = "tab:leaderboard",
) -> str:
    """Format aggregated `mean ± std` metrics as a publication-ready LaTeX `booktabs` table."""
    col_spec = "ll" + "c" * len(metrics_cols)
    escaped_cols = [c.replace("_", r"\_") for c in metrics_cols]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Task & Method & " + " & ".join(escaped_cols) + r" \\",
        r"\midrule",
    ]
    for key, stats in summary.items():
        task, method = key.split("::", 1)
        cells = [task, method.replace("_", r"\_")]
        for m in metrics_cols:
            if m in stats:
                mu = stats[m]["mean"]
                sd = stats[m]["std"]
                n = int(stats[m]["n"])
                cells.append(f"${mu:.4f} \\pm {sd:.4f}$" if n > 1 else f"${mu:.4f}$")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate evaluate.py JSONs into NeurIPS booktabs LaTeX tables.")
    parser.add_argument("--scores-dir", type=Path, required=True, help="Directory containing evaluate.py *.json files.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr_db", "ssim", "fsc_0p5_cyc_per_vox", "d_eff_rel_err", "tortuosity_rel_err"],
        help="Metric keys to include as columns.",
    )
    parser.add_argument("--out-tex", type=Path, default=None)
    args = parser.parse_args(argv)

    files = sorted(args.scores_dir.rglob("*.json"))
    summary = aggregate_score_jsons(files)
    tex = format_booktabs_table(summary, args.metrics)
    if args.out_tex is not None:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text(tex + "\n", encoding="utf-8")
    print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
