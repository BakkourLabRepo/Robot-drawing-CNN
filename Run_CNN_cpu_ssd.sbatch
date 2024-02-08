#!/bin/bash
#SBATCH --job-name=CNN_robot_task_cpu
#SBATCH --account=ssd
#SBATCH --output=CNN_robot_task_cpu.out
#SBATCH --error=CNN_robot_task_cpu.err
#SBATCH --time=00:10:00
#SBATCH --partition=ssd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

module unload python
module load python/anaconda-2022.05
source activate tf

python Run_CNN_Model.py cpu