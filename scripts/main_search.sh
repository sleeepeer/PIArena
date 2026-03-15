#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.out

#SBATCH --job-name=pytorch
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=48:00:00


DATASET=$1
BACKEND_LLM=$2
ATTACK=$3
DEFENSE=$4
NAME=$5
SEED=$6
LOG_FILE=$7
ATTACKER_LLM=$8
BATCH_SIZE=$9

source ~/.bashrc
conda activate piarena
export HF_HOME=Your_HF_Cache_Path
export HF_TOKEN=Your_HF_Token
export OPENAI_API_KEY=Your_OpenAI_API_Key

python3 -u main_search.py \
  --dataset "${DATASET}" \
  --backend_llm "${BACKEND_LLM}" \
  --attack "${ATTACK}" \
  --defense "${DEFENSE}" \
  --name "${NAME}" \
  --seed "${SEED}" \
  --attacker_llm "${ATTACKER_LLM}" \
  --batch_size "${BATCH_SIZE}" \
  > "${LOG_FILE}" 2>&1
