#!/usr/bin/env bash
# 36-Run SLURM GPU Array for Headline Experiment (Task T1 on Primary Verified 2.026x Track)
# Covers 4 training regimes (synthetic_clean, synthetic_noisy, synthetic_jittered, real_paired)
# x 3 OpenLSR-X v0.2.1 models (srresnet_25d, swinir_25d_l1, openlsrx_swinir_physics) x 3 seeds = 36 runs.
# Pass INCLUDE_EXPLORATORY_8X=1 (--array=0-71) only if running the exploratory 8.101x track as well.
#SBATCH --job-name=zn13x_t1_sr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-35
#SBATCH --output=logs/slurm_t1_%A_%a.out

set -euo pipefail
mkdir -p logs outputs/learned_baselines

EXTRA_FLAG=""
if [[ "${INCLUDE_EXPLORATORY_8X:-0}" == "1" ]]; then
  EXTRA_FLAG="--include-exploratory-8x"
fi

CMD=$(python -c "
import json, subprocess, sys
cmd_args = [
    sys.executable, '-m', 'baselines.learned',
    '--suite', 'openlsrx',
    '--config', 'configs/baselines/openlsrx_t1.yaml'
]
if '${EXTRA_FLAG}':
    cmd_args.append('${EXTRA_FLAG}')
out = subprocess.check_output(cmd_args)
cmds = json.loads(out)['commands']
idx = int('${SLURM_ARRAY_TASK_ID:-0}')
print(cmds[idx])
")

echo "[Task ${SLURM_ARRAY_TASK_ID}] Executing: ${CMD}"
eval "${CMD}"
