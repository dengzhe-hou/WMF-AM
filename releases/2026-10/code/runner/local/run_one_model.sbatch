#!/bin/bash
#SBATCH --job-name=wmfam-open
#SBATCH --partition=batch
#SBATCH --nodelist=node-a
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/path/to/repo/results/open/_slurm/slurm-%j.out
# One small open model on one node-a MIG 4g.40gb slice: run protocol steps 5B-5F via localrun/procedure.sh.
# Outputs: /path/to/repo/results/open/<label>/
#
#   sbatch run_one_model.sbatch LABEL
#   sbatch --array=1-N%2 run_one_model.sbatch LABEL_1 ... LABEL_N    (array task i runs LABEL_i)
#   DRY_RUN=1 bash run_one_model.sbatch LABEL                         (login node; no GPU, no server)
#
# Resources
#   gres   4g.40gb slice, the profile Job N ran on; every small model uses the same profile.
#   mem    32G. Job N (granite-3.1-8b, 15.2 GiB weights) peaked at 20.7 GB including the
#          weights in page cache. The largest small model is falcon3-10b (19.2 GiB), so about 25 GB.
#   time   2:30 retained as the existing job limit. Job N took 9 min end to end,
#          with block G capped at 16 tokens (G: 35 s). Option A again caps G at 16;
#          only the 30 minimal prompts gain the explicit final-number instruction.
if [ -n "${SLURM_ARRAY_TASK_ID-}" ]; then
  LABEL=${!SLURM_ARRAY_TASK_ID}
else
  LABEL=${1:?usage: sbatch run_one_model.sbatch LABEL}
fi
TP=1
GPU_UTIL=0.90          # section 5B, models 1 to 10
READY_TIMEOUT_MIN=20   # Job N was ready in 80 s
EXPECT_NGPU=1
# shellcheck source=procedure.sh
source /path/to/repo/localrun/procedure.sh
