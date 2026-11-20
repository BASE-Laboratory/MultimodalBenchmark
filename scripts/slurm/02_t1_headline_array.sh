#!/usr/bin/env bash
# 54-Run SLURM GPU Array for Headline Experiment (Task T1: Synthetic-Trained vs Real-Trained SR)
# Covers 2 tracks (2p026x, 8p101x) x 3 training regimes (synthetic_clean, synthetic_noisy, real_paired)
# x 3 models (unet3d, swinir3d, openlsrx) x 3 seeds (42, 137, 2026) = 54 runs.
# Usage: sbatch scripts/slurm/02_t1_headline_array.sh
#SBATCH --job-name=zn13x_t1_sr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-53
#SBATCH --output=logs/slurm_t1_%A_%a.out

set -euo pipefail
mkdir -p logs outputs/scores/t1

CMD=$(python -c "
import json, subprocess, sys
out = subprocess.check_output([
    sys.executable, '-m', 'baselines.learned',
    '--suite', 'openlsrx',
    '--config', 'configs/baselines/openlsrx_t1.yaml'
])
cmds = json.loads(out)['commands']
idx = int('${SLURM_ARRAY_TASK_ID:-0}')
print(cmds[idx])
")

echo "[Task ${SLURM_ARRAY_TASK_ID}] Executing: ${CMD}"
eval "${CMD}"
