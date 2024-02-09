#!/bin/bash
#SBATCH --job-name=CNN_robot_task_gpu # job name
#SBATCH --account=pi-bakkour
#SBATCH --output=CNN_robot_task_gpu.out # output file
#SBATCH --error=CNN_robot_task_gpu.err # error file
#SBATCH --time=00:10:00 # runtime
#SBATCH --partition=gpu # gpu partition
#SBATCH --mem-per-gpu=30G # mem per GPU
#SBATCH --gres=gpu:2 # Number of GPUs requested

module unload python cuda cudnn
module load python/anaconda-2022.05
module load cuda/11.2
module load cudnn/11.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/software/cuda-11.2-el8-x86_64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/sbgrid/x86_64-linux/careless/0.2.7/lib/python3.8/site-packages/tensorrt/

python Run_CNN_Model.py gpu 