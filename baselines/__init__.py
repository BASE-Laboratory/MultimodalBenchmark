"""Classical and learned baseline implementations & adapters for Tasks T1-T4."""

from .classical import (
    bicubic_chemical_upsample,
    cnmf_coupled_fusion,
    fbp_sparse_reconstruct_2d,
    guided_filter_chemical_upsample,
    guided_tv_chemical_fusion,
    joint_bilateral_upsample,
    lanczos_upsample_3d,
    multi_otsu_watershed_3d,
    sirt_reconstruct_2d,
    tricubic_upsample_3d,
    tv_deconv_upsample_3d,
)

__all__ = [
    "tricubic_upsample_3d",
    "lanczos_upsample_3d",
    "tv_deconv_upsample_3d",
    "bicubic_chemical_upsample",
    "guided_filter_chemical_upsample",
    "joint_bilateral_upsample",
    "guided_tv_chemical_fusion",
    "cnmf_coupled_fusion",
    "fbp_sparse_reconstruct_2d",
    "sirt_reconstruct_2d",
    "multi_otsu_watershed_3d",
]
