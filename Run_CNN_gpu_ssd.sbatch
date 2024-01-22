#!/bin/bash
#SBATCH --job-name=Robot_Drawing_CNN
#SBATCH --account=ssd
#SBATCH --output=Robot_Drawing_CNN.out
#SBATCH --error=Robot_Drawing_CNN.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu # gpu partition
#SBATCH --mem-per-gpu=5G # mem per GPU        
#SBATCH --gres=gpu:1 # Number of GPUs requested

module unload python cudnn cuda tensorflow
module unload gcc/13.2.0
module load cudnn/11.2 
module load cuda/12.2
module load python/anaconda-2023.09
python Run_CNN_Model.py gpu ssd