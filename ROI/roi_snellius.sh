#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=01:00:00

source monai-det/bin/activate # activate your virtual environment
# pip install --no-cache-dir-r requirements.txt

srun python ./roi_detection.py