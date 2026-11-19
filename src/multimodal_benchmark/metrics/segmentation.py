"""T4 phase segmentation metrics: per-class Dice, morphological Boundary F1, and phase-fraction error."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def compute_dice_per_class(
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    class_ids: Sequence[int] = (0, 1, 2),
    mask: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    p = np.asarray(pred_labels, dtype=np.int32)
    r = np.asarray(ref_labels, dtype=np.int32)
    valid = (mask > 0) if mask is not None else np.ones(r.shape, dtype=bool)
    scores: Dict[int, float] = {}
    for cid in class_ids:
        pm = (p == cid) & valid
        rm = (r == cid) & valid
        denom = int(np.count_nonzero(pm) + np.count_nonzero(rm))
        if denom == 0:
            scores[int(cid)] = 1.0
        else:
            inter = int(np.count_nonzero(pm & rm))
            scores[int(cid)] = float(2.0 * inter / denom)
    return scores


def _extract_boundary(binary_mask: np.ndarray) -> np.ndarray:
    m = np.asarray(binary_mask, dtype=bool)
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    return m ^ binary_erosion(m, iterations=1, border_value=0)


def compute_boundary_f1(
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    class_ids: Sequence[int] = (0, 1, 2),
    tolerance_vox: int = 1,
) -> float:
    """Compute macro-averaged Boundary F1 within a morphological tolerance of `tolerance_vox` voxels."""
    p = np.asarray(pred_labels, dtype=np.int32)
    r = np.asarray(ref_labels, dtype=np.int32)
    f1s = []
    for cid in class_ids:
        bp = _extract_boundary(p == cid)
        br = _extract_boundary(r == cid)
        np_b = int(np.count_nonzero(bp))
        nr_b = int(np.count_nonzero(br))
        if np_b == 0 and nr_b == 0:
            f1s.append(1.0)
            continue
        if np_b == 0 or nr_b == 0:
            f1s.append(0.0)
            continue
        br_dil = binary_dilation(br, iterations=max(1, int(tolerance_vox)))
        bp_dil = binary_dilation(bp, iterations=max(1, int(tolerance_vox)))
        prec = float(np.count_nonzero(bp & br_dil)) / np_b
        rec = float(np.count_nonzero(br & bp_dil)) / nr_b
        f1s.append(float(2.0 * prec * rec / (prec + rec + 1e-12)))
    return float(np.mean(f1s)) if f1s else 0.0


def compute_phase_fraction_error(
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    class_ids: Sequence[int] = (0, 1, 2),
    mask: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    p = np.asarray(pred_labels, dtype=np.int32)
    r = np.asarray(ref_labels, dtype=np.int32)
    valid = (mask > 0) if mask is not None else np.ones(r.shape, dtype=bool)
    n_tot = max(1, int(np.count_nonzero(valid)))
    errs: Dict[int, float] = {}
    for cid in class_ids:
        fp = float(np.count_nonzero((p == cid) & valid)) / n_tot
        fr = float(np.count_nonzero((r == cid) & valid)) / n_tot
        errs[int(cid)] = abs(fp - fr)
    return errs


def compute_segmentation_metrics(
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    class_names: Optional[Mapping[int, str]] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute complete T4a/T4b segmentation metrics dictionary."""
    if class_names is None:
        unique_ids = sorted(int(v) for v in np.union1d(np.unique(ref_labels), np.unique(pred_labels)))
        class_names = {cid: f"class_{cid}" for cid in unique_ids}
    cids = list(class_names.keys())

    dice = compute_dice_per_class(pred_labels, ref_labels, class_ids=cids, mask=mask)
    pf_err = compute_phase_fraction_error(pred_labels, ref_labels, class_ids=cids, mask=mask)
    bf1_1 = compute_boundary_f1(pred_labels, ref_labels, class_ids=cids, tolerance_vox=1)
    bf1_2 = compute_boundary_f1(pred_labels, ref_labels, class_ids=cids, tolerance_vox=2)

    out: Dict[str, float] = {
        "macro_dice": float(np.mean(list(dice.values()))),
        "boundary_f1_tol1": bf1_1,
        "boundary_f1_tol2": bf1_2,
        "mean_phase_fraction_mae": float(np.mean(list(pf_err.values()))),
    }
    for cid, cname in class_names.items():
        slug = cname.lower().replace("-", "_").replace(" ", "_")
        out[f"dice_{slug}"] = dice[cid]
        out[f"phase_fraction_err_{slug}"] = pf_err[cid]
    return out
