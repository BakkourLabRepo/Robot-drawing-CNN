#!/bin/bash
#SBATCH --job-name=Robot_Drawing_CNN
#SBATCH --account=ssd
#SBATCH --output=Robot_Drawing_CNN.out
#SBATCH --error=Robot_Drawing_CNN.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=40G # memory per CPU
#SBATCH --partition=ssd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

module unload python cudnn cuda tensorflow
module unload gcc/13.2.0
module load cudnn/11.2 
module load cuda/12.2
module load python/anaconda-2023.09
python Run_CNN_Model.py cpu ssd