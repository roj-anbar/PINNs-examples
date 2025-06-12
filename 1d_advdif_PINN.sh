#!/bin/bash
#
#SBATCH --job-name=1d_advdif_PINN
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00

# Requesting resources
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#
# Export all environment vars
#SBATCH --export=ALL

# Load modules
module load StdEnv/2023 python/3.10.13 scipy-stack cuda/12.6

# Activate the environment
source ~/PINNs/bin/activate

# Navigate to the script directory
cd /scratch/ranbar/PINNs/Examples/scripts

# Run the training
python 1d_advdif_PINN.py
