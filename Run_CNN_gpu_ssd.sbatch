#!/bin/bash
#SBATCH --job-name=CNN_robot_task_gpu # job name
#SBATCH --account=pi-bakkour
#SBATCH --output=CNN_robot_task_gpu.out # output file
#SBATCH --error=CNN_robot_task_gpu.err # error file
#SBATCH --time=01:00:00 # runtime
#SBATCH --partition=gpu # gpu partition
#SBATCH --mem-per-gpu=30G # mem per GPU
#SBATCH --gres=gpu:2 # Number of GPUs requested

source /software/tensorflow-el8-x86_64/2.8.4-gcc-8.4.1/tfenv
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/software/cuda-11.2-el8-x86_64

python Run_CNN_Model.py --core GPU --environment midway --model_name VGG19