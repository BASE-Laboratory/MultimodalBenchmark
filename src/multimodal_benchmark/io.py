"""Out-of-core chunked loaders for synchrotron XCT (.h5), DIAD XRD-CT (.nxs), and Rigaku ASCII XRD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

XCT_DATASET_CANDIDATES: Tuple[str, ...] = (
    "4-TomopyRecon-tomo/data",
    "4-AstraReconGpu-tomo/data",
    "entry/final_result_tomo/data",
    "data",
    "volume",
)


@dataclass(frozen=True)
class PowderXRDSpectrum:
    two_theta_deg: np.ndarray
    q_inv_angstrom: np.ndarray
    intensity: np.ndarray
    wavelength_angstrom: float = 1.54059


@dataclass(frozen=True)
class XRDCTSinogram:
    scan_id: Optional[int]
    data: np.ndarray          # shape: (n_q, n_angles, n_translations)
    q_inv_angstrom: np.ndarray
    angles_deg: np.ndarray
    kb_cs_x_px: Optional[np.ndarray] = None
    kb_cs_y_px: Optional[float] = None


def two_theta_to_q(two_theta_deg: np.ndarray, wavelength_angstrom: float = 1.54059) -> np.ndarray:
    """Convert diffraction angle 2*theta (deg) to momentum transfer q = 4*pi*sin(theta)/lambda (A^-1)."""
    theta_rad = np.deg2rad(np.asarray(two_theta_deg, dtype=np.float64) * 0.5)
    return (4.0 * np.pi / wavelength_angstrom) * np.sin(theta_rad)


def load_ascii_xrd(path: Path | str, wavelength_angstrom: float = 1.54059) -> PowderXRDSpectrum:
    """Parse Rigaku SmartLab .asc two-theta/omega file into calibrated 2*theta, q, and intensity arrays."""
    path = Path(path)
    start_deg: Optional[float] = None
    step_deg: Optional[float] = None
    intensities: List[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("*START"):
                start_deg = float(line.split("=")[-1].strip())
            elif line.startswith("*STEP"):
                step_deg = float(line.split("=")[-1].strip())
            elif line.startswith("*"):
                continue
            else:
                for tok in line.replace(",", " ").split():
                    try:
                        intensities.append(float(tok))
                    except ValueError:
                        break

    if start_deg is None or step_deg is None or not intensities:
        raise ValueError(f"Could not parse Rigaku .asc header/intensities from {path}")

    y = np.asarray(intensities, dtype=np.float64)
    two_theta = start_deg + step_deg * np.arange(y.size, dtype=np.float64)
    q = two_theta_to_q(two_theta, wavelength_angstrom=wavelength_angstrom)
    return PowderXRDSpectrum(two_theta_deg=two_theta, q_inv_angstrom=q, intensity=y, wavelength_angstrom=wavelength_angstrom)


def find_xct_dataset_path(h5_file: h5py.File) -> str:
    """Locate the primary 3D XCT dataset inside a Savu TomoPy, Astra, or cropped HDF5 file."""
    for candidate in XCT_DATASET_CANDIDATES:
        if candidate in h5_file and isinstance(h5_file[candidate], h5py.Dataset):
            return candidate

    found: Optional[str] = None

    def _visitor(name: str, obj: object) -> None:
        nonlocal found
        if found is None and isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            found = name

    h5_file.visititems(_visitor)
    if found is None:
        raise KeyError("No 3D tomography dataset found in HDF5 file.")
    return found


def _is_savu_xzy_layout(ds_name: str, shape: Tuple[int, ...]) -> bool:
    if len(shape) != 3:
        return False
    if any(tag in ds_name for tag in ("TomopyRecon", "AstraRecon")):
        return shape[0] == shape[2] and shape[1] != shape[0]
    return shape in ((2510, 2110, 2510), (2560, 2160, 2560))


def get_xct_shape_and_dtype(
    path: Path | str,
    dataset_path: Optional[str] = None,
    canonical_zyx: bool = True,
) -> Tuple[Tuple[int, int, int], np.dtype, str]:
    with h5py.File(path, "r") as h5f:
        ds_name = dataset_path or find_xct_dataset_path(h5f)
        ds = h5f[ds_name]
        raw_shape = tuple(int(s) for s in ds.shape)
        if canonical_zyx and _is_savu_xzy_layout(ds_name, raw_shape):
            # Savu stores (X, Z_vertical, Y) -> canonical (Z, Y, X) is (shape[1], shape[2], shape[0])
            raw_shape = (raw_shape[1], raw_shape[2], raw_shape[0])
        return raw_shape, ds.dtype, ds_name


def load_xct_subvolume(
    path: Path | str,
    zyx_slices: Optional[Tuple[slice, slice, slice]] = None,
    stride: int = 1,
    flip_x: bool = False,
    dataset_path: Optional[str] = None,
    dtype: Optional[np.dtype] = np.float32,
    canonical_zyx: bool = True,
) -> np.ndarray:
    """Load a 3D XCT sub-volume out-of-core in canonical (Z, Y, X) order without reading the full >50 GB array."""
    with h5py.File(path, "r") as h5f:
        ds_name = dataset_path or find_xct_dataset_path(h5f)
        ds = h5f[ds_name]
        is_xzy = canonical_zyx and _is_savu_xzy_layout(ds_name, tuple(int(s) for s in ds.shape))
        if zyx_slices is None:
            sl_zyx = (slice(None, None, stride), slice(None, None, stride), slice(None, None, stride))
        else:
            sz, sy, sx = zyx_slices
            sl_zyx = (
                slice(sz.start, sz.stop, sz.step or stride),
                slice(sy.start, sy.stop, sy.step or stride),
                slice(sx.start, sx.stop, sx.step or stride),
            )
        if is_xzy:
            # Disk layout is (X, Z, Y); slice out-of-core then transpose (1, 2, 0) -> (Z, Y, X)
            raw_sl = (sl_zyx[2], sl_zyx[0], sl_zyx[1])
            arr = np.transpose(np.asarray(ds[raw_sl]), (1, 2, 0))
        else:
            arr = np.asarray(ds[sl_zyx])
    if flip_x:
        arr = np.flip(arr, axis=2).copy()
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def load_xrdct_sinogram(
    path: Path | str,
    q_range: Optional[Tuple[float, float]] = None,
    scan_id: Optional[int] = None,
) -> XRDCTSinogram:
    """Load DIAD XRD-CT integrated diffraction sinogram (q, angle, translation), actual motor angles, and KB XCT coords."""
    path = Path(path)
    with h5py.File(path, "r") as h5f:
        q = np.asarray(h5f["processed/result/q"][:], dtype=np.float64).squeeze()
        ds = h5f["processed/result/data"]
        if q_range is not None:
            mask = (q >= q_range[0]) & (q <= q_range[1])
            idx = np.where(mask)[0]
            if idx.size == 0:
                raise ValueError(f"No q bins inside range {q_range}")
            q_sl = slice(int(idx[0]), int(idx[-1]) + 1)
            q = q[q_sl]
        else:
            q_sl = slice(None)

        # DAWN MovingBeamAzimuthalIntegration stores (1, n_angles, n_trans, n_q) with q last;
        # legacy synthetic/transposed files may store (n_q, n_angles, n_trans) with q first.
        if ds.ndim == 4 and ds.shape[-1] >= q.size:
            raw = np.moveaxis(np.asarray(ds[0, :, :, q_sl], dtype=np.float32), -1, 0)
        elif ds.ndim == 3 and ds.shape[-1] >= q.size and ds.shape[0] != ds.shape[-1]:
            raw = np.moveaxis(np.asarray(ds[:, :, q_sl], dtype=np.float32), -1, 0)
        else:
            raw = np.asarray(ds[q_sl], dtype=np.float32)

        n_angles = raw.shape[1]
        angles = None
        for ang_path in ("processed/result/gts_cs_theta", "entry/diffraction_sum/gts_cs_theta"):
            if ang_path in h5f:
                angles = np.asarray(h5f[ang_path][:], dtype=np.float64).ravel()[:n_angles]
                break
        if angles is None:
            angles = np.linspace(0.0, 360.0, n_angles, endpoint=False, dtype=np.float64)

        kb_cs_x = (
            np.asarray(h5f["processed/result/kb_cs_x"][:], dtype=np.float64).ravel()
            if "processed/result/kb_cs_x" in h5f
            else None
        )
        kb_cs_y = (
            float(np.asarray(h5f["processed/result/kb_cs_y"][:]).ravel()[0])
            if "processed/result/kb_cs_y" in h5f
            else None
        )

    return XRDCTSinogram(
        scan_id=scan_id,
        data=raw,
        q_inv_angstrom=q,
        angles_deg=angles,
        kb_cs_x_px=kb_cs_x,
        kb_cs_y_px=kb_cs_y,
    )


def load_xrdct_peak_maps(
    path: Path | str,
    flip_x: bool = False,
) -> Dict[float, np.ndarray]:
    """Load all reconstructed peak maps ('entry/peak at q~...') from an output_<scan>_{na,zn}.nxs file."""
    peaks: Dict[float, np.ndarray] = {}
    with h5py.File(path, "r") as h5f:
        grp = h5f["entry"]
        for key in grp.keys():
            if key.startswith("peak at q~"):
                q_val = float(key.split("~", 1)[1])
                img = np.asarray(grp[key][:], dtype=np.float32).squeeze()
                if flip_x:
                    img = np.fliplr(img).copy()
                peaks[q_val] = img
    if not peaks:
        raise KeyError(f"No 'entry/peak at q~*' datasets found in {path}")
    return dict(sorted(peaks.items(), key=lambda kv: kv[0]))


def aggregate_peak_maps(
    peaks: Dict[float, np.ndarray],
    mode: str = "all_sum",
    target_q: Optional[float] = None,
    q_tol: float = 0.015,
) -> np.ndarray:
    """Combine multi-reflection XRD-CT peak maps via sum, mean, low-q/high-q subset, or single peak."""
    ordered = list(peaks.items())
    if mode == "single_peak":
        if target_q is None:
            raise ValueError("target_q is required when mode='single_peak'")
        matches = [arr for q, arr in ordered if abs(q - target_q) <= q_tol]
        if not matches:
            raise KeyError(f"No reflection within {q_tol} A^-1 of q={target_q}")
        return matches[0].astype(np.float32)
    if mode == "all_sum":
        return np.sum([arr for _, arr in ordered], axis=0).astype(np.float32)
    if mode == "all_mean":
        return np.mean([arr for _, arr in ordered], axis=0).astype(np.float32)
    if mode == "low_q":
        half = max(1, len(ordered) // 2)
        return np.sum([arr for _, arr in ordered[:half]], axis=0).astype(np.float32)
    if mode == "high_q":
        half = max(1, len(ordered) // 2)
        return np.sum([arr for _, arr in ordered[half:]], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def load_xrdct_volume_50um(
    data_dir: Path | str,
    phase: str,
    scan_ids: Optional[Sequence[int]] = None,
    aggregation: str = "all_sum",
    flip_x: bool = False,
    correct_legacy_47p5_deg_rotation: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """Assemble the 3D (Z, 20, 20) 50 um XRD-CT volume from output_<scan>_{na,zn}.nxs files.

    Note: Between the 25 um ROI scans (k11-43322..43325, `gts_cs_theta = [0, 360]` deg) and the 50 um volume scans
    (k11-43336..43401, `gts_cs_theta = [-47.5, +312.5]` deg), the `gts_cs_theta` soft-motor zero was offset by -47.5 deg
    while starting at the same physical sample orientation (mean-subtracted sinogram correlation r = 0.9956 between
    k11-43324 and k11-43345 at frame 0). Because `DiffTomo_Analysis.ipynb` passed `deg2rad(gts_cs_theta)` directly to
    `tomopy.recon`, every pre-reconstructed `output_43336..43400_{na,zn}.nxs` slice is rotated by -47.5 deg relative to
    the 25 um ROI slices. Setting `correct_legacy_47p5_deg_rotation=True` (default) applies the +47.5 deg correction
    (-47.5 deg when `flip_x=True`).
    """
    from scipy.ndimage import rotate as nd_rotate

    phase_lc = phase.lower()
    if phase_lc not in ("na", "zn"):
        raise ValueError("phase must be 'na' or 'zn'")
    ddir = Path(data_dir)
    if scan_ids is None:
        default_ids = list(range(43336, 43348)) + list(range(43393, 43401))
        scan_ids = [s for s in default_ids if (ddir / f"output_{s}_{phase_lc}.nxs").exists()]
    slices: List[np.ndarray] = []
    used_ids: List[int] = []
    target_q = 1.651 if phase_lc == "na" else 1.656
    rot_angle = -47.5 if flip_x else 47.5
    for sid in scan_ids:
        fpath = ddir / f"output_{sid}_{phase_lc}.nxs"
        peaks = load_xrdct_peak_maps(fpath, flip_x=flip_x)
        sl = aggregate_peak_maps(peaks, mode=aggregation, target_q=target_q)
        if correct_legacy_47p5_deg_rotation:
            sl = nd_rotate(sl, angle=rot_angle, reshape=False, order=1, mode="nearest").astype(np.float32)
        slices.append(sl)
        used_ids.append(int(sid))
    if not slices:
        raise FileNotFoundError(f"No 50 um output_*_{phase_lc}.nxs files found in {ddir}")
    return np.stack(slices, axis=0), used_ids
