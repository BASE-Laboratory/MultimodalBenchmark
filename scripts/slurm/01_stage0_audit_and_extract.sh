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

echo "[4/4] Extracting Two-Track float32 sub-volumes (raw_float32 + cleaned_dering_v1 + difference maps)..."
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

echo "Stage 0A-2, Stage 0C, and Two-Track float32 extraction completed."
