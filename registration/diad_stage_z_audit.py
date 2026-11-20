"""Stage 0A-2: Standalone DLS K11 raw NeXus vertical stage-z motor audit for 25 um <-> 50 um slice matching.

On Zenodo (`13329639`), DAWN 2.33.0 sinogram exports retain `/processed` (`kb_cs_y = 1080.0` px) and omit raw `/entry`
vertical stage encoders. On SCARF / Diamond Light Source (`/dls/k11/data/2024/mg32980-1/nexus`), this script reads
`k11-43322..43325.nxs` (25 um) and `k11-43336..43405.nxs` (50 um), resolves the active vertical translation axis, and
verifies the axial slice mapping `{43324: z=9 (43345), 43322: z=14 (43395), 43323: z=17 (43398), 43325: z=18 (43399)}`
including explicit margin diagnostics on `43324` (`43345` vs `43346`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from registration.diad_correlative import read_dls_k11_raw_stage_z

DEFAULT_SCANS_25UM: Tuple[int, ...] = (43324, 43322, 43323, 43325)
DEFAULT_SCANS_50UM: Tuple[int, ...] = tuple(list(range(43336, 43348)) + list(range(43393, 43402)))
CORRELATION_MATCHED_50UM: Dict[int, Tuple[int, int]] = {
    43324: (9, 43345),
    43322: (14, 43395),
    43323: (17, 43398),
    43325: (18, 43399),
}


def select_active_vertical_motor(
    scans_meta: Mapping[str, Mapping[str, float]],
    scans_50um: Sequence[int] = DEFAULT_SCANS_50UM,
) -> Optional[str]:
    """Identify the NeXus motor path that steps monotonically across the 50 um axial XRD-CT series."""
    present_50 = [str(s) for s in scans_50um if str(s) in scans_meta]
    if len(present_50) < 3:
        return None

    candidate_keys: set[str] = set()
    for sid in present_50:
        candidate_keys.update(scans_meta[sid].keys())

    best_key: Optional[str] = None
    best_span = 0.0
    for k in sorted(candidate_keys):
        vals = [scans_meta[sid][k] for sid in present_50 if k in scans_meta[sid]]
        if len(vals) < max(3, len(present_50) // 2):
            continue
        span = float(np.max(vals) - np.min(vals))
        # Active vertical stage steps ~50 um (0.05 mm) per scan across ~20 scans (~1.0 mm or ~1000 um span)
        if span > best_span and any(tok in k.lower() for tok in ("y", "z", "height", "gts", "sam")):
            diffs = np.diff(vals)
            nonzero = diffs[np.abs(diffs) > 1e-9]
            if nonzero.size > 0:
                sign_consistency = abs(float(np.mean(np.sign(nonzero))))
                score = span * (0.5 + 0.5 * sign_consistency)
                if score > best_span:
                    best_span = score
                    best_key = k
    return best_key


def resolve_stage_z_matches(
    raw_stage_report: Mapping[str, Any],
    scans_25um: Sequence[int] = DEFAULT_SCANS_25UM,
    scans_50um: Sequence[int] = DEFAULT_SCANS_50UM,
    expected_correlation_matches: Mapping[int, Tuple[int, int]] = CORRELATION_MATCHED_50UM,
) -> Dict[str, Any]:
    """Match each 25 um scan to its nearest 50 um scan by physical stage motor position."""
    if not raw_stage_report.get("available", False):
        return {
            "status": "PENDING_SCARF_EXECUTION",
            "available": False,
            "dls_raw_nexus_dir": raw_stage_report.get("dls_raw_nexus_dir"),
            "note": (
                "Raw DLS K11 NeXus files (/dls/k11/data/2024/mg32980-1/nexus/k11-*.nxs) not mounted in current "
                "environment. Run this script on SCARF/Hamilton before finalizing Table 3."
            ),
        }

    scans_meta: Mapping[str, Mapping[str, float]] = raw_stage_report["scans"]
    motor_key = select_active_vertical_motor(scans_meta, scans_50um=scans_50um)
    if motor_key is None:
        return {
            "status": "NO_VARYING_VERTICAL_MOTOR_FOUND",
            "available": True,
            "scans_inspected": len(scans_meta),
        }

    pos_50: List[Tuple[int, int, float]] = []
    for z_idx, sid50 in enumerate(scans_50um):
        if str(sid50) in scans_meta and motor_key in scans_meta[str(sid50)]:
            pos_50.append((z_idx, int(sid50), float(scans_meta[str(sid50)][motor_key])))

    # Infer unit scale (mm vs um): adjacent 50 um scans step by ~0.05 in mm or ~50.0 in um
    if len(pos_50) >= 2:
        med_step_raw = float(np.median(np.abs(np.diff([p[2] for p in pos_50]))))
    else:
        med_step_raw = 0.05
    unit_to_um = 1000.0 if med_step_raw < 2.0 else 1.0

    matches: Dict[str, Any] = {}
    all_agree = True
    for sid25 in scans_25um:
        if str(sid25) not in scans_meta or motor_key not in scans_meta[str(sid25)]:
            matches[str(sid25)] = {"status": "MISSING_25UM_NEXUS"}
            all_agree = False
            continue
        z25 = float(scans_meta[str(sid25)][motor_key])
        ranked = sorted(pos_50, key=lambda t: abs(t[2] - z25))
        best_idx, best_sid50, best_z50 = ranked[0]
        runner_idx, runner_sid50, runner_z50 = ranked[1] if len(ranked) > 1 else ranked[0]

        exp_idx, exp_sid50 = expected_correlation_matches.get(int(sid25), (-1, -1))
        agrees = int(best_sid50) == int(exp_sid50)
        if not agrees:
            all_agree = False

        matches[str(sid25)] = {
            "scan_25um": int(sid25),
            "stage_pos_raw": z25,
            "matched_50um_z_index": int(best_idx),
            "matched_50um_scan": int(best_sid50),
            "offset_to_best_um": round(abs(best_z50 - z25) * unit_to_um, 3),
            "runner_up_50um_z_index": int(runner_idx),
            "runner_up_50um_scan": int(runner_sid50),
            "offset_to_runner_up_um": round(abs(runner_z50 - z25) * unit_to_um, 3),
            "correlation_matched_scan": int(exp_sid50),
            "agrees_with_correlation_match": bool(agrees),
        }

    return {
        "status": "VERIFIED_ALL_AGREE" if all_agree else "MISMATCH_DETECTED_UPDATE_SPLITS",
        "available": True,
        "active_vertical_motor": motor_key,
        "inferred_unit_to_um": unit_to_um,
        "median_50um_step_um": round(med_step_raw * unit_to_um, 3),
        "matches": matches,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 0A-2: DLS K11 raw NeXus vertical stage-z slice audit.")
    parser.add_argument(
        "--dls-raw-nexus-dir",
        type=Path,
        default=Path("/dls/k11/data/2024/mg32980-1/nexus"),
        help="Path to raw Diamond K11 NeXus directory containing k11-43322..43401.nxs",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/stage0a2_diad_stage_z_audit.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args(argv)

    all_ids = list(DEFAULT_SCANS_25UM) + list(DEFAULT_SCANS_50UM)
    raw_rep = read_dls_k11_raw_stage_z(args.dls_raw_nexus_dir, scan_ids=all_ids)
    resolved = resolve_stage_z_matches(raw_rep)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(resolved, fh, indent=2)
    print(json.dumps(resolved, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
