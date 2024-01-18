#!/bin/bash
#SBATCH --job-name=Drawing_CNN
#SBATCH --account=ssd
#SBATCH --output=Drawing_CNN.out
#SBATCH --error=Drawing_CNN.err
#SBATCH --time=24:00:00
#SBATCH --partition=ssd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

python Run_CNN_Model.py

