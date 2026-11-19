"""Drop-in 7-DOF registration & non-cupping adapter for BASE-Laboratory/OpenLSR-X (`openlsrx.data.zn13x`).

Upgrades OpenLSR-X's integer-only `register_multires_pair_3d` with:
1. Full 7-DOF SimpleITK similarity registration (6-DOF rigid + isotropic objective magnification scale factor).
2. Automated 3D macropore/inclusion centroid Target Registration Error (TRE in um).
3. Cylindrical local-tomography validity masking for the truncated 0.325 um (20x) scan.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from registration.i13_ladder import (
    LadderRegistrationResult,
    radial_non_cupping_mask,
    register_3d_similarity,
    resample_moving_to_fixed_zyx,
)


def align_openlsrx_7dof_pair(
    lr_zyx: np.ndarray,
    hr_zyx: np.ndarray,
    lr_voxel_um: float,
    hr_voxel_um: float,
    apply_local_tomo_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Register LR volume to HR volume via 7-DOF similarity transform and return aligned LR-on-HR grid + metadata."""
    hr = np.asarray(hr_zyx, dtype=np.float32)
    lr = np.asarray(lr_zyx, dtype=np.float32)
    hr_sp = (float(hr_voxel_um), float(hr_voxel_um), float(hr_voxel_um))
    lr_sp = (float(lr_voxel_um), float(lr_voxel_um), float(lr_voxel_um))

    mask = radial_non_cupping_mask(hr.shape, valid_radius_fraction=0.82) if apply_local_tomo_mask else None

    reg_res: LadderRegistrationResult = register_3d_similarity(
        fixed_zyx=hr,
        moving_zyx=lr,
        fixed_spacing_zyx_um=hr_sp,
        moving_spacing_zyx_um=lr_sp,
        fixed_mask_zyx=mask,
        metric="mattes_mi",
        allow_scale=True,
    )

    lr_resampled_on_hr = resample_moving_to_fixed_zyx(
        moving_zyx=lr,
        moving_spacing_zyx_um=lr_sp,
        fixed_shape_zyx=hr.shape,
        fixed_spacing_zyx_um=hr_sp,
        transform=reg_res.transform,
    )
    return lr_resampled_on_hr, hr, reg_res.to_dict()
