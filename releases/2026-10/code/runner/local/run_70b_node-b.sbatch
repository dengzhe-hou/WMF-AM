#!/bin/bash
#SBATCH --job-name=wmfam-70b
#SBATCH --partition=batch
#SBATCH --nodelist=node-b
#SBATCH --gres=gpu:pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=170G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/repo/results/open/_slurm/slurm-%j.out
# llama-3.3-70b-instruct on node-b, both RTX PRO 6000 cards, tensor parallel 2: run protocol steps 5B-5F
# (model 11 command: --tensor-parallel-size 2, --gpu-memory-utilization 0.95) via localrun/procedure.sh.
# CUDA_VISIBLE_DEVICES comes from SLURM. Outputs: results/open/llama-3.3-70b-instruct/
#
#   sbatch run_70b_node-b.sbatch                  (only when both cards are free; weights must be downloaded)
#   DRY_RUN=1 bash run_70b_node-b.sbatch          (login node; no GPU, no server)
#
# Job N stalled in the first collective before loading weights. The bounded probes in
# profile_output/comm_20261003 reproduce the P2P and custom-all-reduce stalls; NCCL SHM works.
# New attempts must set RESULTS_ROOT to a fresh directory and --output to its _slurm directory.
# procedure.sh runs the original pre-stated check after the server is ready and
# before the smoke test: 2 GPUs each with memory.used >= 61440 MiB, the engine line says
# tensor_parallel_size=2, and a compute process on each GPU UUID. If it fails, nothing is run.
#
# Resources
#   gres   pro_6000:2. 141.1 GB (131.4 GiB) of bf16 weights, about 65.7 GiB per card; each card has
#          95.6 GiB; at 0.95 about 20 GiB per card remain for KV cache and CUDA graphs.
#   cpus   8 of node-b's 16: API server, engine core, two GPU workers, torch.compile workers.
#   mem    170G. vLLM 0.30.0 prefetches a checkpoint on NFS into page cache when it fits in 90 % of
#          "available RAM". It reads /sys/fs/cgroup/memory.max, which the SLURM job cgroup does not
#          expose here (Job N logged "Available RAM: 462.79 GiB" under --mem=32G), so it sees
#          node-b's host RAM and will prefetch all 131.4 GiB. That page cache is charged to the job.
#          Processes need about 25-30 GiB more (Job N: 20.7 GB peak with 15.2 GiB of weights, so
#          about 5 GB per single-GPU server; TP=2 adds a second worker and compile workers).
#          131.4 + 30 = about 160 GiB. 170G is below node-b's grantable 181.5 GiB (192000 MB minus
#          MemSpecLimit 6144 MB). A smaller limit would not OOM (page cache is reclaimable) but would
#          evict prefetched pages and read the 141 GB from NFS a second time.
#   time   6:00 retained as the existing job limit; TP=2 throughput on node-b is unmeasured.
#          Option A caps all 150 G items at 16 tokens (2400 output tokens in total at
#          the cap). Block I still allows 512 per item. Loading is allowed up to 1 h.
LABEL=llama-3.3-70b-instruct
TP=2
GPU_UTIL=0.95          # section 5B, model 11
READY_TIMEOUT_MIN=60   # prompt allows 30 min for model 11; here 131 GiB come from NFS on node-b
EXPECT_NGPU=2
export NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO
SERVE_EXTRA_ARGS=(--disable-custom-all-reduce)
# shellcheck source=procedure.sh
source /path/to/repo/localrun/procedure.sh
