#!/bin/bash
#SBATCH --job-name=Precursors
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1         # Request H100 GPU
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --output=logs/job_out_%A_%a.log
#SBATCH --error=logs/job_err_%A_%a.log
           

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate my_env_310

nvidia-smi --query-gpu=name,memory.total --format=csv


echo ">>> Running job $SLURM_ARRAY_TASK_ID"


tau=15
lookback=$((4 * tau))
label=$((2 * tau))



srun python -u main.py \
    --model_architecture Transformer-based \
    --inp FTLE \
    --tau "$tau" \
    --lookback "$lookback" \
    --label "$label" 
    #--predict_model


