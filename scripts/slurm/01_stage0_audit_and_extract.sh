#!/usr/bin/env bash
# Turnkey SCARF / Hamilton Stage 0A-2, Stage 0C, 32-bit 7-DOF Refit, Two-Track Extraction & Baseline Prep Job
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
T1_BENCH_DIR="${T1_BENCH_DIR:-data/benchmarks/zn13x_float32}"
T2_BENCH_DIR="${T2_BENCH_DIR:-data/benchmarks/zn13x_t2_diad}"

mkdir -p logs outputs "${DERIVED_OUT_DIR}" "${T1_BENCH_DIR}" "${T2_BENCH_DIR}"

echo "[1/6] Running Stage 0A-2: DLS K11 raw NeXus vertical stage-z slice audit..."
python -m registration.diad_stage_z_audit \
  --dls-raw-nexus-dir "${DLS_K11_NEXUS_DIR}" \
  --out-json outputs/stage0a2_diad_stage_z_audit.json

echo "[2/6] Running Stage 0A-2: DIAD 0.54 um XCT <-> 25/50 um XRD-CT chirality & z-height audit..."
python -m registration.diad_correlative \
  --diad-xct-h5 "${ZENODO_FLOAT32_DIR}/k11-43334_tomo_p4_astra_recon_gpu.h5" \
  --xrdct-dir "${ZENODO_FLOAT32_DIR}/xrdct" \
  --dls-raw-nexus-dir "${DLS_K11_NEXUS_DIR}" \
  --out-json outputs/stage0a_diad_correlative.json \
  --qc-png outputs/stage0a_diad_correlative_qc.png

echo "[3/6] Running Stage 0C: Cross-beamline (I13-2 169067 <-> DIAD k11-43334) fiducial-vs-pellet & octant audit..."
python -m registration.cross_beamline_audit \
  --i13-h5 "${ZENODO_FLOAT32_DIR}/169067_tomo_p4_tomopy_recon.h5" \
  --diad-h5 "${ZENODO_FLOAT32_DIR}/k11-43334_tomo_p4_astra_recon_gpu.h5" \
  --out-yaml outputs/stage0c_cross_beamline_regions.yaml \
  --qc-png outputs/stage0c_cross_beamline_qc.png

echo "[4/6] Refitting I13-2 7-DOF similarity transforms directly on 32-bit float32 Savu reconstructions..."
python -m registration.i13_ladder \
  --h5-dir "${ZENODO_FLOAT32_DIR}" \
  --z0 640 \
  --z1 1920 \
  --stride 4 \
  --out-json outputs/i13_float32_refitted_transforms.json \
  --out-certified-yaml configs/i13_certified_regions.yaml

echo "[5/6] Extracting Two-Track paired float32 sub-volumes (primary_raw_float32 + cleaned_dering_v1 + diff maps)..."
python -m preprocessing.extract_benchmark_tracks \
  --lr-h5 "${ZENODO_FLOAT32_DIR}/169067_tomo_p4_tomopy_recon.h5" \
  --hr-h5 "${ZENODO_FLOAT32_DIR}/169066_tomo_p4_tomopy_recon.h5" \
  --track 2p026x \
  --lr-vox-um 1.625 \
  --hr-vox-um 0.8125 \
  --cal-hr-vox-um 0.8006 \
  --z0 640 \
  --z1 1920 \
  --out-h5 "${DERIVED_OUT_DIR}/t1_2p026x_tracks.h5"

if [[ "${INCLUDE_EXPLORATORY_8X:-0}" == "1" ]]; then
  python -m preprocessing.extract_benchmark_tracks \
    --lr-h5 "${ZENODO_FLOAT32_DIR}/169068_tomo_p4_tomopy_recon.h5" \
    --hr-h5 "${ZENODO_FLOAT32_DIR}/169065_tomo_p4_tomopy_recon.h5" \
    --track 8p101x \
    --lr-vox-um 2.600 \
    --hr-vox-um 0.325 \
    --cal-hr-vox-um 0.3211 \
    --z0 640 \
    --z1 1920 \
    --out-h5 "${DERIVED_OUT_DIR}/t1_8p101x_tracks.h5"
fi

echo "[6/6] Materializing split-sliced TIFF & calibrated .ome.zarr baseline inputs for 02_t1 & 03_t2 SLURM arrays..."
python -m preprocessing.prepare_baseline_inputs \
  --mode t1 \
  --extracted-h5 "${DERIVED_OUT_DIR}/t1_2p026x_tracks.h5" \
  --track 2p026x \
  --cal-hr-vox-um 0.8006 \
  --out-dir "${T1_BENCH_DIR}"

if [[ "${INCLUDE_EXPLORATORY_8X:-0}" == "1" ]]; then
  python -m preprocessing.prepare_baseline_inputs \
    --mode t1 \
    --extracted-h5 "${DERIVED_OUT_DIR}/t1_8p101x_tracks.h5" \
    --track 8p101x \
    --cal-hr-vox-um 0.3211 \
    --out-dir "${T1_BENCH_DIR}"
fi

python -m preprocessing.prepare_baseline_inputs \
  --mode t2 \
  --xrdct-dir "${ZENODO_FLOAT32_DIR}/xrdct" \
  --diad-xct-h5 "${ZENODO_FLOAT32_DIR}/k11-43334_tomo_p4_astra_recon_gpu.h5" \
  --out-dir "${T2_BENCH_DIR}"

echo "Stage 0 audits, 32-bit 7-DOF refit, Two-Track extraction, and T1/T2 baseline input preparation completed."
