#!/bin/bash -l

#SBATCH --job-name=train_model_%a
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=0
#SBATCH --output /home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data/scripts/outerr/outerr_train_model.%A_%a.out  # Name of stdout output file (%A expands to jobId and %a expands to the array index), remember to make outerr directory
#SBATCH --error /home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data/scripts/outerr/outerr_train_model.%A_%a.err  # Name of stdout output file (%A expands to jobId and %a expands to the array index), remember to make outerr directory
#SBATCH --mail-user=cguerra3@jhu.edu
#SBATCH --mail-type=ALL

# Module loads
module load cuda
module load python/3.6

pip install --user torch torchvision
pip install --user h5py
pip install --user matplotlib
pip install --user adabound
pip install --user tqdm
pip install --user biopython

date
# Add project to path
export PYTHONPATH="$PYTHONPATH:/home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction"

export DATA_DIR=/home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data/casp11_h5

export SCRIPT=/home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/cli/pnet_trainer_cli.py
time python $SCRIPT $DATA_DIR --num_blocks1D 3 --num_blocks2D 25 --batch_size 1 --models_dir /home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/models --save_every 300 --max_training_seq_len 450 --max_validation_seq_len 450
date

