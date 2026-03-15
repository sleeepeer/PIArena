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
ATTACK_PATH=$5
NAME=$6
SEED=$7
LOG_FILE=$8

source ~/.bashrc
conda activate piarena
export HF_HOME=Your_HF_Cache_Path
export HF_TOKEN=Your_HF_Token
export OPENAI_API_KEY=Your_OpenAI_API_Key

python3 -u main.py \
  --dataset "${DATASET}" \
  --backend_llm "${BACKEND_LLM}" \
  --attack "${ATTACK}" \
  --defense "${DEFENSE}" \
  --attack_path "${ATTACK_PATH}" \
  --name "${NAME}" \
  --seed "${SEED}" \
  > "${LOG_FILE}" 2>&1
