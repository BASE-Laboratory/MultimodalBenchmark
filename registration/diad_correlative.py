"""Stage 0A (Part 2): Correlative DIAD XCT <-> XRD-CT chirality, axial z-height, and 2D similarity alignment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, percentile_filter, uniform_filter1d
from scipy.optimize import minimize


@dataclass(frozen=True)
class CorrelativeAlignmentResult:
    best_z_center_vox: float
    best_z_center_um: float
    flip_x: bool
    shift_yx_xrd_px: Tuple[float, float]
    shift_yx_um: Tuple[float, float]
    rotation_deg: float
    scale: float
    nmi_score: float
    pearson_r: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalized_mutual_information_2d(
    img_a: np.ndarray,
    img_b: np.ndarray,
    bins: int = 32,
) -> float:
    """Compute Normalized Mutual Information NMI(A, B) = (H(A) + H(B)) / H(A, B) >= 1.0."""
    a = np.asarray(img_a, dtype=np.float64).ravel()
    b = np.asarray(img_b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 16 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return 1.0

    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist_2d / (np.sum(hist_2d) + 1e-15)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    nz_x = px > 0
    nz_y = py > 0
    nz_xy = pxy > 0
    hx = -np.sum(px[nz_x] * np.log(px[nz_x]))
    hy = -np.sum(py[nz_y] * np.log(py[nz_y]))
    hxy = -np.sum(pxy[nz_xy] * np.log(pxy[nz_xy]))
    return float((hx + hy) / (hxy + 1e-15))


def blur_xct_to_xrdct_grid(
    xct_zyx: np.ndarray,
    xct_voxel_um: float = 0.54,
    xrd_pixel_um: float = 25.0,
    beam_thickness_um: Optional[float] = None,
    target_yx_shape: Optional[Tuple[int, int]] = None,
    fov_center_yx_vox: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Apply physical 3D pencil-beam PSF (axial slab thickness + in-plane spot blur) and resample onto XRD-CT yx grid.

    For DIAD 0.54 um XCT:
      - 25 um XRD-CT averages over ~46.3 axial XCT slices per slice.
      - 50 um XRD-CT averages over ~92.6 axial XCT slices per slice.
    """
    vol = np.asarray(xct_zyx, dtype=np.float32)
    nz, ny, nx = vol.shape
    slab_um = float(beam_thickness_um if beam_thickness_um is not None else xrd_pixel_um)
    slab_vox = max(1, int(np.rint(slab_um / xct_voxel_um)))
    inplane_sigma_vox = max(0.5, (xrd_pixel_um / xct_voxel_um) / 2.355)

    # 1. Axial boxcar integration along z (pencil-beam vertical height)
    if slab_vox > 1 and nz > 1:
        vol_z = uniform_filter1d(vol, size=min(slab_vox, nz), axis=0, mode="nearest")
    else:
        vol_z = vol

    # 2. Sample onto target (ny_xrd, nx_xrd) grid at physical step = xrd_pixel_um / xct_voxel_um
    step_vox = xrd_pixel_um / xct_voxel_um
    if target_yx_shape is None:
        ny_out = max(4, int(np.floor(ny / step_vox)))
        nx_out = max(4, int(np.floor(nx / step_vox)))
    else:
        ny_out, nx_out = target_yx_shape

    if fov_center_yx_vox is None:
        cy, cx = 0.5 * (ny - 1), 0.5 * (nx - 1)
    else:
        cy, cx = fov_center_yx_vox

    out = np.zeros((nz, ny_out, nx_out), dtype=np.float32)
    offs_y = (np.arange(ny_out, dtype=np.float64) - 0.5 * (ny_out - 1)) * step_vox + cy
    offs_x = (np.arange(nx_out, dtype=np.float64) - 0.5 * (nx_out - 1)) * step_vox + cx

    # Pre-filter in-plane slice-by-slice and bilinearly sample onto coarse grid
    yi0 = np.clip(np.floor(offs_y).astype(np.int32), 0, ny - 1)
    yi1 = np.clip(yi0 + 1, 0, ny - 1)
    wy = (offs_y - yi0).astype(np.float32)[:, None]

    xi0 = np.clip(np.floor(offs_x).astype(np.int32), 0, nx - 1)
    xi1 = np.clip(xi0 + 1, 0, nx - 1)
    wx = (offs_x - xi0).astype(np.float32)[None, :]

    for iz in range(nz):
        sl = gaussian_filter(vol_z[iz], sigma=inplane_sigma_vox, mode="nearest")
        v00 = sl[np.ix_(yi0, xi0)]
        v01 = sl[np.ix_(yi0, xi1)]
        v10 = sl[np.ix_(yi1, xi0)]
        v11 = sl[np.ix_(yi1, xi1)]
        out[iz] = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)

    return out


def _warp_2d_similarity(
    img: np.ndarray,
    dy_px: float,
    dx_px: float,
    rot_deg: float,
    scale: float,
) -> np.ndarray:
    """Apply 2D similarity transform around image center mapping output coords -> input coords."""
    ny, nx = img.shape
    center = np.array([0.5 * (ny - 1), 0.5 * (nx - 1)], dtype=np.float64)
    theta = np.deg2rad(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    inv_s = 1.0 / max(scale, 1e-3)
    # Output coord y_out = center + scale * R * (y_in - center) + shift => y_in = center + inv_s * R^T * (y_out - shift - center)
    mat = inv_s * np.array([[c, s], [-s, c]], dtype=np.float64)
    shift = np.array([dy_px, dx_px], dtype=np.float64)
    offset = center - mat @ (center + shift)
    return affine_transform(img, matrix=mat, offset=offset, order=1, mode="nearest")


def align_xrdct_slice_to_diad_xct(
    xrd_total_yx: np.ndarray,
    blurred_xct_zyx: np.ndarray,
    xct_voxel_um: float = 0.54,
    xrd_pixel_um: float = 25.0,
    z_search_range: Optional[Tuple[int, int]] = None,
    test_both_chiralities: bool = True,
) -> CorrelativeAlignmentResult:
    """Register a 2D total-scattering XRD-CT slice to an anisotropically PSF-blurred DIAD XCT stack.

    Explicitly tests both x-chiralities (flip_x in {False, True}) to resolve the Savu AstraReconGpu
    vs TomoPy gridrec mirror convention, finds the optimal axial z-height, and refines sub-pixel
    2D similarity (dy, dx, rotation_deg, scale) via Normalized Mutual Information.
    """
    target = np.asarray(xrd_total_yx, dtype=np.float64)
    stack = np.asarray(blurred_xct_zyx, dtype=np.float64)
    if stack.shape[1:] != target.shape:
        raise ValueError(f"Blurred XCT yx shape {stack.shape[1:]} must match XRD-CT slice {target.shape}")

    nz = stack.shape[0]
    z0, z1 = z_search_range if z_search_range is not None else (0, nz)
    z0 = max(0, int(z0))
    z1 = min(nz, int(z1))
    chiralities = (False, True) if test_both_chiralities else (False,)

    best_coarse = (-1.0, z0, False, 0, 0)
    # Coarse integer shift + axial z + chirality sweep
    max_int_shift = max(2, min(5, target.shape[0] // 5))
    for flip_x in chiralities:
        for iz in range(z0, z1):
            cand = np.fliplr(stack[iz]) if flip_x else stack[iz]
            for dy in range(-max_int_shift, max_int_shift + 1):
                for dx in range(-max_int_shift, max_int_shift + 1):
                    moved = _warp_2d_similarity(cand, float(dy), float(dx), 0.0, 1.0)
                    score = normalized_mutual_information_2d(target, moved, bins=24)
                    if score > best_coarse[0]:
                        best_coarse = (score, iz, flip_x, dy, dx)

    _, z_init, best_flip, dy_init, dx_init = best_coarse

    def _interp_z_slice(z_float: float) -> np.ndarray:
        zc = float(np.clip(z_float, 0.0, nz - 1.0))
        i0 = int(np.floor(zc))
        i1 = min(nz - 1, i0 + 1)
        w = zc - i0
        sl = (1.0 - w) * stack[i0] + w * stack[i1]
        return np.fliplr(sl) if best_flip else sl

    # Continuous Powell refinement over (z, dy, dx, rot_deg, scale)
    x0 = np.array([float(z_init), float(dy_init), float(dx_init), 0.0, 1.0], dtype=np.float64)
    bounds = [
        (float(max(0, z_init - 5)), float(min(nz - 1, z_init + 5))),
        (dy_init - 3.0, dy_init + 3.0),
        (dx_init - 3.0, dx_init + 3.0),
        (-8.0, 8.0),
        (0.92, 1.08),
    ]

    def _obj(params: np.ndarray) -> float:
        zf, dyf, dxf, rotf, scf = params
        if not (bounds[0][0] <= zf <= bounds[0][1] and bounds[4][0] <= scf <= bounds[4][1]):
            return 10.0
        sl = _interp_z_slice(zf)
        warped = _warp_2d_similarity(sl, dyf, dxf, rotf, scf)
        # Combine NMI with slight correlation tie-breaker for smooth sub-pixel convergence
        nmi = normalized_mutual_information_2d(target, warped, bins=24)
        if np.std(warped) > 1e-12 and np.std(target) > 1e-12:
            r = float(np.corrcoef(target.ravel(), warped.ravel())[0, 1])
        else:
            r = 0.0
        return -(nmi + 0.15 * r)

    opt = minimize(_obj, x0, method="Powell", bounds=bounds, options={"maxiter": 250, "ftol": 1e-4})
    z_opt, dy_opt, dx_opt, rot_opt, sc_opt = [float(v) for v in opt.x]
    final_slice = _warp_2d_similarity(_interp_z_slice(z_opt), dy_opt, dx_opt, rot_opt, sc_opt)
    final_nmi = normalized_mutual_information_2d(target, final_slice, bins=24)
    final_r = float(np.corrcoef(target.ravel(), final_slice.ravel())[0, 1]) if np.std(final_slice) > 1e-12 else 0.0

    return CorrelativeAlignmentResult(
        best_z_center_vox=z_opt,
        best_z_center_um=z_opt * xct_voxel_um,
        flip_x=best_flip,
        shift_yx_xrd_px=(dy_opt, dx_opt),
        shift_yx_um=(dy_opt * xrd_pixel_um, dx_opt * xrd_pixel_um),
        rotation_deg=rot_opt,
        scale=sc_opt,
        nmi_score=final_nmi,
        pearson_r=final_r,
    )


def read_savu_or_zyx_slab(
    h5_path: Path | str,
    z0: int = 0,
    z1: Optional[int] = None,
    z_stride: int = 1,
    yx_stride: int = 1,
    dataset_path: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int], str]:
    """Read a memory-safe `(Z, Y, X)` slab out-of-core via `basecoat.open_savu_h5`."""
    from basecoat import open_savu_h5

    with open_savu_h5(h5_path, dataset_path=dataset_path) as lv:
        full_zyx = (int(lv.shape[0]), int(lv.shape[1]), int(lv.shape[2]))
        ds_name = str(lv.provenance["dataset_path"])
        z_end = full_zyx[0] if z1 is None else min(full_zyx[0], int(z1))
        vol = lv.read_region(
            (slice(z0, z_end, z_stride), slice(None, None, yx_stride), slice(None, None, yx_stride))
        )
    return np.asarray(vol.data, dtype=np.float32), full_zyx, ds_name


def read_dls_k11_raw_stage_z(
    dls_raw_nexus_dir: Path | str,
    scan_ids: Sequence[int] = (
        43322,
        43323,
        43324,
        43325,
        43336,
        43344,
        43345,
        43346,
        43393,
        43394,
        43395,
        43396,
        43397,
        43398,
        43399,
        43400,
    ),
) -> Dict[str, Any]:
    """Read raw /entry stage height motors from /dls/k11/data/2024/mg32980-1/nexus/k11-*.nxs on SCARF/Diamond."""
    ndir = Path(dls_raw_nexus_dir)
    if not ndir.exists():
        return {"available": False, "reason": f"Directory {ndir} not mounted in current environment."}

    motor_candidates = [
        "entry/diffraction/gts_y",
        "entry/diffraction/gts_z",
        "entry/diffraction/samy",
        "entry/diffraction/sam_y",
        "entry/instrument/gts/gts_y",
        "entry/instrument/gts/gts_z",
        "entry/before_scan/gts/gts_y",
        "entry/before_scan/gts/gts_z",
        "entry/before_scan/sample_stage/samy",
    ]
    per_scan: Dict[str, Dict[str, float]] = {}
    for sid in scan_ids:
        p = ndir / f"k11-{sid}.nxs"
        if not p.exists():
            continue
        vals: Dict[str, float] = {}
        with h5py.File(p, "r") as fh:
            for mpath in motor_candidates:
                if mpath in fh and isinstance(fh[mpath], h5py.Dataset):
                    arr = np.asarray(fh[mpath][()], dtype=np.float64)
                    if arr.size > 0 and np.all(np.isfinite(arr)):
                        vals[mpath] = float(np.mean(arr))
            if not vals and "entry" in fh:
                def _scan_scalars(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset) and obj.size <= 4 and obj.dtype.kind in "fi":
                        if any(tok in name.lower() for tok in ("gts", "sam", "stage", "kb", "height", "_y", "_z")):
                            arr = np.asarray(obj[()], dtype=np.float64)
                            if arr.size > 0 and np.all(np.isfinite(arr)):
                                vals[f"entry/{name}"] = float(np.mean(arr))

                fh["entry"].visititems(_scan_scalars)
        per_scan[str(sid)] = vals

    return {"available": bool(per_scan), "dls_raw_nexus_dir": str(ndir), "scans": per_scan}


def _reconstruct_total_bragg_slice_from_nxs(
    nxs_path: Path,
    apply_50um_theta_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, float]:
    from .xrdct_reference import estimate_conjugate_cor, reconstruct_sinogram_2d

    with h5py.File(nxs_path, "r") as fh:
        q = np.squeeze(fh["processed/result/q"][:])
        d = np.squeeze(fh["processed/result/data"][:])
        th = np.squeeze(fh["processed/result/gts_cs_theta"][:]) + float(apply_50um_theta_offset_deg)
        kb_x = np.squeeze(fh["processed/result/kb_cs_x"][:])
    q_ax = [i for i, s in enumerate(d.shape) if s == q.size][0]
    d = np.moveaxis(d, q_ax, 0)
    if d.shape[1] < d.shape[2]:
        d = np.swapaxes(d, 1, 2)
    qmask = (q >= 0.85) & (q <= 3.10)
    sub = d[qmask].astype(np.float64)
    bg = percentile_filter(sub, percentile=15, size=(45, 1, 1))
    bragg_sino = np.sum(np.clip(sub - bg, 0.0, None), axis=0)
    cor = estimate_conjugate_cor(bragg_sino, th)
    rec = reconstruct_sinogram_2d(bragg_sino, th, center_px=cor.center_px)
    return rec, float(np.mean(kb_x))


def run_diad_correlative_cli_audit(
    diad_xct_h5: Path,
    xrdct_dir: Path,
    out_json: Path,
    qc_png: Optional[Path] = None,
    dls_raw_nexus_dir: Optional[Path] = None,
    dataset_path: Optional[str] = None,
    xct_voxel_um: float = 0.54,
    z_stride: int = 4,
    yx_stride: int = 2,
) -> Dict[str, Any]:
    """Stream-downsample DIAD 0.54 um XCT HDF5 and align 25 um & 50 um XRD-CT slices testing both x-chiralities."""
    xct_sub, full_zyx, ds_name = read_savu_or_zyx_slab(
        diad_xct_h5, z0=0, z1=None, z_stride=z_stride, yx_stride=yx_stride, dataset_path=dataset_path
    )
    eff_yx_vox_um = xct_voxel_um * yx_stride
    eff_z_vox_um = xct_voxel_um * z_stride

    # Calibrated kb_cs_x center is 1280.50 px in full 2560-px DIAD coordinates; scale if file is smaller test volume
    scale_x = full_zyx[2] / 2560.0
    scale_y = full_zyx[1] / 2560.0
    fov_center_sub = (
        0.5 * (xct_sub.shape[1] - 1) if abs(scale_y - 1.0) > 0.1 else (1280.5 / yx_stride),
        0.5 * (xct_sub.shape[2] - 1) if abs(scale_x - 1.0) > 0.1 else (1280.5 / yx_stride),
    )

    results_25um: Dict[str, Any] = {}
    first_pair_vis: Optional[Tuple[np.ndarray, np.ndarray, str]] = None

    for sid in (43322, 43323, 43324, 43325):
        p25 = xrdct_dir / f"k11-{sid}-diffraction-DiffInt_Mask_Azz_Txt.nxs"
        if not p25.exists():
            continue
        rec25, _ = _reconstruct_total_bragg_slice_from_nxs(p25, apply_50um_theta_offset_deg=0.0)
        blurred_25 = blur_xct_to_xrdct_grid(
            xct_sub,
            xct_voxel_um=eff_yx_vox_um,
            xrd_pixel_um=25.0,
            beam_thickness_um=25.0 * (eff_yx_vox_um / eff_z_vox_um),
            target_yx_shape=rec25.shape,
            fov_center_yx_vox=fov_center_sub,
        )
        aln = align_xrdct_slice_to_diad_xct(
            xrd_total_yx=rec25,
            blurred_xct_zyx=blurred_25,
            xct_voxel_um=eff_z_vox_um,
            xrd_pixel_um=25.0,
            test_both_chiralities=True,
        )
        d = aln.to_dict()
        d["best_full_xct_z_vox"] = round(aln.best_z_center_vox * z_stride, 2)
        results_25um[str(sid)] = d
        if first_pair_vis is None:
            iz = int(np.clip(round(aln.best_z_center_vox), 0, blurred_25.shape[0] - 1))
            sl = np.fliplr(blurred_25[iz]) if aln.flip_x else blurred_25[iz]
            warped = _warp_2d_similarity(sl, aln.shift_yx_xrd_px[0], aln.shift_yx_xrd_px[1], aln.rotation_deg, aln.scale)
            first_pair_vis = (rec25, warped, f"k11-{sid} (flip_x={aln.flip_x}, r={aln.pearson_r:.3f})")

    raw_stage_meta = (
        read_dls_k11_raw_stage_z(dls_raw_nexus_dir)
        if dls_raw_nexus_dir is not None
        else {"available": False, "reason": "--dls-raw-nexus-dir not specified"}
    )

    if qc_png is not None and first_pair_vis is not None:
        from .cross_beamline_audit import save_checkerboard_qc_png

        save_checkerboard_qc_png(first_pair_vis[0], first_pair_vis[1], qc_png, tile_px=4)

    payload: Dict[str, Any] = {
        "diad_xct_h5": str(diad_xct_h5),
        "dataset_path": ds_name,
        "full_volume_shape_zyx": list(full_zyx),
        "stream_strides_zyx": [z_stride, yx_stride, yx_stride],
        "scans_25um_alignment": results_25um,
        "dls_raw_nexus_stage_z_audit": raw_stage_meta,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 0A-2: Correlative DIAD 0.54 um XCT <-> 25/50 um XRD-CT chirality & z-height alignment."
    )
    parser.add_argument("--diad-xct-h5", type=Path, required=True, help="Path to tomo_p4_astra_recon_gpu.h5")
    parser.add_argument("--xrdct-dir", type=Path, default=Path("data/cache/zenodo"), help="Directory with k11-*.nxs")
    parser.add_argument(
        "--dls-raw-nexus-dir",
        type=Path,
        default=Path("/dls/k11/data/2024/mg32980-1/nexus"),
        help="Optional Diamond/SCARF raw NeXus directory containing /entry vertical stage motor positions.",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional HDF5 dataset path override.")
    parser.add_argument("--z-stride", type=int, default=4, help="Axial stride when streaming 57 GB HDF5 (default: 4).")
    parser.add_argument("--yx-stride", type=int, default=2, help="In-plane stride when streaming 57 GB HDF5 (default: 2).")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/stage0a_diad_correlative.json"))
    parser.add_argument("--qc-png", type=Path, default=Path("outputs/stage0a_diad_correlative_qc.png"))
    args = parser.parse_args(argv)

    rep = run_diad_correlative_cli_audit(
        diad_xct_h5=args.diad_xct_h5,
        xrdct_dir=args.xrdct_dir,
        out_json=args.out_json,
        qc_png=args.qc_png,
        dls_raw_nexus_dir=args.dls_raw_nexus_dir,
        dataset_path=args.dataset_path,
        z_stride=args.z_stride,
        yx_stride=args.yx_stride,
    )
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

