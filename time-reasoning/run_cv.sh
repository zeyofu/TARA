#!/bin/bash
#SBATCH --job-name=ben-cv
#SBATCH --output=/shared/xzhou45/time-reasoning/slurm_output.txt
#SBATCH --partition=p_nlp
#SBATCH --nodelist=nlpgpu04
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB

cd /shared/xzhou45/time-reasoning

venv/bin/python run.py \
    --type 2 \
    --to 2 \
    --epochs 25 \
    --name trial \
    --mode train
