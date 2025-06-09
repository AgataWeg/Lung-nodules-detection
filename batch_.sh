#!/bin/bash
#SBATCH --nodes=
#SBATCH --ntasks=
#SBATCH --cpus-per-task=
#SBATCH --gpus=
#SBATCH --partition=
#SBATCH --reservation=
#SBATCH --time=

# activate your virtual environment:
source venv/bin/activate 
# add your script here:
srun python ./your_script.py
