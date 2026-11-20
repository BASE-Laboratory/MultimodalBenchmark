#!/usr/bin/env bash
# 36-Run SLURM GPU Array for Task T2 (X-FUSE v1.0.0 `xfuse fuse` & `xfuse fuse-3d` across 4 LOSO folds)
# Covers 3 X-FUSE variants (xfuse_2d_jbu_gating, xfuse_2d_cross_attn, xfuse_3d_loso_triplane)
# x 4 LOSO folds (43324, 43322, 43323, 43325) x 3 seeds = 36 runs.
# Requires Stage 0A-2 and Stage 0C to have completed first.
#SBATCH --job-name=zn13x_t2_xfuse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-35
#SBATCH --output=logs/slurm_t2_%A_%a.out

set -euo pipefail
mkdir -p logs outputs/learned_baselines

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
