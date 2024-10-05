#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=400G
#SBATCH --partition=isi

source /home1/spangher/.bashrc

source activate hay

start_idx=$1
end_idx=$2

cd /project/jonmay_1426/spangher/conditional-information-retrieval

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/project/jonmay_231/spangher/huggingface_cache

python3 interleaving/interleave_dpr.py --start_idx 0 --end_idx 100 --iterations=20