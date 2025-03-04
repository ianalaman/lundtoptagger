#!/bin/bash

# job name
#SBATCH --job-name=tagger_training

# choose the GPU queue
#SBATCH -p GPU

# request one node
#SBATCH -N1
#SBATCH --exclusive

# keep environment variables
#SBATCH --export=ALL

# request CPUs
#SBATCH -n12

# request 1 A100 GPU
#SBATCH --gres=gpu:a100:1

# request enough memory
#SBATCH --mem=100G

# email notifications
#SBATCH --mail-user=youremail@ucl.ac.uk
#SBATCH --mail-type=ALL

# change log names; %j gives job id, %x gives job name
#SBATCH --output=~/job_outputs/slurm-%j.%x.out
# optional separate error output file
# #SBATCH --error=~/job_outputs/slurm-%j.%x.err

# set max job wall time to 6 days
#SBATCH --time=6-00:00:00

cd ~/Lund_tagging/lundtoptagger
echo "Moved dir, now in:"
pwd

echo "Activating environment"
eval "$(/share/apps/anaconda/3-2022.05/bin/conda shell.bash hook)"
conda activate lundtoptagger
echo $CONDA_DEFAULT_ENV

echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

echo "Running training script..."
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml