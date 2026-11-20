#!/usr/bin/env bash
# 24-Run SLURM GPU Array for Task T2 (4-Fold Leave-One-Slice-Out Chemical Fusion)
# Covers 4 LOSO folds (43324, 43322, 43323, 43325) x 2 models (dual_encoder_unet, xfuse) x 3 seeds = 24 runs.
# Requires Stage 0A-2 and Stage 0C to have completed first.
# Usage: sbatch scripts/slurm/03_t2_loso_array.sh
#SBATCH --job-name=zn13x_t2_loso
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/slurm_t2_%A_%a.out

set -euo pipefail
mkdir -p logs outputs/scores/t2

if [[ ! -f "outputs/stage0c_cross_beamline_regions.yaml" ]]; then
  echo "ERROR: Task T2 XCT-guided fusion is gated until Stage 0C (outputs/stage0c_cross_beamline_regions.yaml) has run." >&2
  exit 1
fi

CMD=$(python -c "
import json, subprocess, sys
out = subprocess.check_output([
    sys.executable, '-m', 'baselines.learned',
    '--suite', 'xfuse',
    '--config', 'configs/baselines/xfuse_t2.yaml'
])
cmds = json.loads(out)['commands']
idx = int('${SLURM_ARRAY_TASK_ID:-0}')
print(cmds[idx])
")

echo "[Task ${SLURM_ARRAY_TASK_ID}] Executing: ${CMD}"
eval "${CMD}"
