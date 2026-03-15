#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.out

#SBATCH --job-name=agentdojo
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=16    
#SBATCH --gpus-per-node=1        
#SBATCH --mem=120g           
#SBATCH --time=48:00:00

# Parameter parsing for main_agentdojo.py
MODEL=$1
ATTACK=$2
DEFENSE=$3
SUITE=$4
TENSOR_PARALLEL_SIZE=$5
NAME=$6
LOG_FILE=$7

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

# Build command arguments
CMD="python -u main_agentdojo.py --model \"${MODEL}\" --attack \"${ATTACK}\" --defense \"${DEFENSE}\" --tensor_parallel_size \"${TENSOR_PARALLEL_SIZE}\" --name \"${NAME}\""

# Add suite if specified (not "all")
if [ "${SUITE}" != "all" ]; then
    CMD="${CMD} --suite \"${SUITE}\""
fi

# Run main_agentdojo.py with output redirected to log file
eval ${CMD} > "${LOG_FILE}" 2>&1
