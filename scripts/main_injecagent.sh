#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.out

#SBATCH --job-name=injecagent
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=16    
#SBATCH --gpus-per-node=1        
#SBATCH --mem=120g           
#SBATCH --time=48:00:00

# Parameter parsing for main_injecagent.py
MODEL=$1
DEFENSE=$2
NAME=$3
SEED=$4
LOG_FILE=$5

set -e

# Environment setup
# module purge
# module load default
# module load gcc/11.4.0
# module load cuda/12.6.1
# module load cuda-compat/12.9

source ~/.bashrc
export HF_HOME=Your_HF_Cache_Path
export HF_TOKEN=Your_HF_Token
export OPENAI_API_KEY=Your_OpenAI_API_Key

# NCCL configuration
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_NET=Socket
# export NCCL_SOCKET_IFNAME=lo
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export PYTHONUNBUFFERED=1

# Run main_injecagent.py with output redirected to log file
python -u main_injecagent.py \
    --model "${MODEL}" \
    --defense "${DEFENSE}" \
    --name "${NAME}" \
    --seed "${SEED}" \
    --checkpoint_interval 10 \
    > "${LOG_FILE}" 2>&1
