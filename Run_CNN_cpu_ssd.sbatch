#!/bin/bash
#SBATCH --job-name=Robot_Drawing_CNN
#SBATCH --account=ssd
#SBATCH --output=Robot_Drawing_CNN.out
#SBATCH --error=Robot_Drawing_CNN.err
#SBATCH --time=24:00:00
#SBATCH --partition=ssd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

module load cudnn
module load python/anaconda-2022.05
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/software/cuda-11.2-el8-x86_64

python Run_CNN_Model.py cpu ssd