#!/usr/bin/env bash
# Turnkey SCARF / Hamilton Stage 0A-2, Stage 0C, and Two-Track Float32 Extraction Job
# Usage: sbatch scripts/slurm/01_stage0_audit_and_extract.sh
#SBATCH --job-name=zn13x_stage0_extract
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_stage0_%j.out

set -euo pipefail

# Override these environment variables for SCARF vs Hamilton mount paths:
DLS_K11_NEXUS_DIR="${DLS_K11_NEXUS_DIR:-/dls/k11/data/2024/mg32980-1/nexus}"
ZENODO_FLOAT32_DIR="${ZENODO_FLOAT32_DIR:-/work/zn13x_zenodo_float32}"
DERIVED_OUT_DIR="${DERIVED_OUT_DIR:-outputs/derived_tracks_float32}"

mkdir -p logs outputs "${DERIVED_OUT_DIR}"

echo "[1/4] Running Stage 0A-2: DLS K11 raw NeXus vertical stage-z slice audit..."
python -m registration.diad_stage_z_audit \
  --dls-raw-nexus-dir "${DLS_K11_NEXUS_DIR}" \
  --out-json outputs/stage0a2_diad_stage_z_audit.json

echo "[2/4] Running Stage 0A-2: DIAD 0.54 um XCT <-> 25/50 um XRD-CT chirality & z-height audit..."
python -m registration.diad_correlative \
  --diad-xct-h5 "${ZENODO_FLOAT32_DIR}/k11-43334_tomo_p4_astra_recon_gpu.h5" \
  --xrdct-dir "${ZENODO_FLOAT32_DIR}/xrdct" \
  --dls-raw-nexus-dir "${DLS_K11_NEXUS_DIR}" \
  --out-json outputs/stage0a_diad_correlative.json \
  --qc-png outputs/stage0a_diad_correlative_qc.png

echo "[3/4] Running Stage 0C: Cross-beamline (I13-2 169067 <-> DIAD k11-43334) fiducial-vs-pellet & octant audit..."
python -m registration.cross_beamline_audit \
  --i13-h5 "${ZENODO_FLOAT32_DIR}/169067_tomo_p4_tomopy_recon.h5" \
  --diad-h5 "${ZENODO_FLOAT32_DIR}/k11-43334_tomo_p4_astra_recon_gpu.h5" \
  --out-yaml outputs/stage0c_cross_beamline_regions.yaml \
  --qc-png outputs/stage0c_cross_beamline_qc.png

echo "[4/5] Extracting Two-Track float32 sub-volumes (raw_float32 + cleaned_dering_v1 + difference maps)..."
for SCAN_SPEC in \
  "169068:2.5945" \
  "169067:1.6212" \
  "169066:0.8006" \
  "169065:0.3211"; do
  SCAN_ID="${SCAN_SPEC%%:*}"
  VOX_UM="${SCAN_SPEC##*:}"
  python -m preprocessing.extract_benchmark_tracks \
    --input "${ZENODO_FLOAT32_DIR}/${SCAN_ID}_tomo_p4_tomopy_recon.h5" \
    --out-dir "${DERIVED_OUT_DIR}/${SCAN_ID}" \
    --voxel-um "${VOX_UM}" \
    --z-min 640 --z-max 1920
done

echo "[5/5] Refitting I13-2 7-DOF ladder transforms directly on 32-bit float32 sub-volumes (upgrading provisional 8-bit preview fits)..."
python -c "
import json, pathlib
import numpy as np, tifffile
from registration.i13_ladder import register_similarity_7dof

out = {}
pairs = [
    ('1p604x', '169068', 2.5945, '169067', 1.6212, 6.5),
    ('2p026x', '169067', 1.6212, '169066', 0.8006, 3.24),
    ('2p493x', '169066', 0.8006, '169065', 0.3211, 1.60),
    ('5p051x', '169067', 1.6212, '169065', 0.3211, 3.24),
]
root = pathlib.Path('${DERIVED_OUT_DIR}')
for name, lr_id, lr_vox, hr_id, hr_vox, work_um in pairs:
    lr = tifffile.imread(root / lr_id / 'primary_raw_float32.tif').astype(np.float32)
    hr = tifffile.imread(root / hr_id / 'primary_raw_float32.tif').astype(np.float32)
    res = register_similarity_7dof(hr, hr_vox, lr, lr_vox, work_voxel_um=work_um)
    out[name] = res.to_dict()
pathlib.Path('outputs/i13_float32_refitted_transforms.json').write_text(json.dumps(out, indent=2))
print('Wrote outputs/i13_float32_refitted_transforms.json')
"

echo "Stage 0A-2, Stage 0C, Two-Track float32 extraction, and 32-bit 7-DOF refit completed."
