"""Tests for Savu (X, Z, Y) -> (Z, Y, X) out-of-core slicing and X-FUSE / OpenLSR-X integration adapters."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from baselines.integrations import align_openlsrx_7dof_pair, load_xfuse_benchmark_t3_bundle
from multimodal_benchmark.io import get_xct_shape_and_dtype, load_xct_subvolume


def test_savu_xzy_out_of_core_transposition(tmp_path: Path) -> None:
    h5_path = tmp_path / "mock_savu.h5"
    # Savu layout: (X=20, Z_vertical=12, Y=20)
    raw_xzy = np.arange(20 * 12 * 20, dtype=np.float32).reshape(20, 12, 20)
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("4-AstraReconGpu-tomo/data", data=raw_xzy)

    shape_zyx, _, ds_name = get_xct_shape_and_dtype(h5_path)
    assert ds_name == "4-AstraReconGpu-tomo/data"
    assert shape_zyx == (12, 20, 20)

    sub = load_xct_subvolume(h5_path, zyx_slices=(slice(2, 6), slice(4, 10), slice(5, 13)))
    expected = np.transpose(raw_xzy[5:13, 2:6, 4:10], (1, 2, 0))
    np.testing.assert_allclose(sub, expected)


@pytest.mark.skipif(
    not Path("data/cache/zenodo/output_43324_zn.nxs").exists(),
    reason="Cached Zenodo XRD-CT outputs not present",
)
def test_xfuse_adapter_loads_real_zenodo_t3_bundle() -> None:
    bundle = load_xfuse_benchmark_t3_bundle("data/cache/zenodo")
    assert bundle["guide_50um_zyx"]["Na"].shape == (20, 20, 20)
    assert bundle["guide_50um_zyx"]["Zn"].shape == (20, 20, 20)
    assert set(bundle["held_out_test_25um"].keys()) == {43322, 43323, 43324, 43325}
    # Verify guard-band mask holds out the majority of adjacent 50 um slices
    assert 1 <= int(np.sum(bundle["train_self_sup_mask_50um"])) <= 10


def test_openlsrx_7dof_adapter_aligns_synthetic_pair() -> None:
    zz, yy, xx = np.indices((24, 32, 32), dtype=np.float32)
    hr = (
        np.exp(-0.5 * ((zz - 8) ** 2 + (yy - 10) ** 2 + (xx - 11) ** 2) / 6.0)
        + 1.5 * np.exp(-0.5 * ((zz - 16) ** 2 + (yy - 21) ** 2 + (xx - 19) ** 2) / 7.0)
        + 1.2 * np.exp(-0.5 * ((zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 22) ** 2) / 5.0)
    )
    lr = hr[::2, ::2, ::2].copy()
    lr_on_hr, hr_out, meta = align_openlsrx_7dof_pair(lr, hr, lr_voxel_um=1.625, hr_voxel_um=0.8125)
    assert lr_on_hr.shape == hr_out.shape == (24, 32, 32)
    assert 0.94 < meta["transform"]["scale"] < 1.06
