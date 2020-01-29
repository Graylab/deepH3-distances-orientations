#!/bin/bash -l

#SBATCH --job-name=train_model_debug_%a
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=15GB
#SBATCH --output /home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data/scripts/outerr/outerr_train_model_debug.%A_%a.out  # Name of stdout output file (%A expands to jobId and %a expands to the array index), remember to make outerr directory
#SBATCH --error /home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data/scripts/outerr/outerr_train_model_debug.%A_%a.err  # Name of stdout output file (%A expands to jobId and %a expands to the array index), remember to make outerr directory
#SBATCH --mail-user=cguerra3@jhu.edu
#SBATCH --mail-type=ALL

# Module loads
# module load cuda
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
time python $SCRIPT $DATA_DIR --num_blocks1D 1 --num_blocks2D 1 --batch_size 1 --models_dir /home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/models --save_every 2 --max_training_iter 5 --max_validation_iter 3 --epochs 20
date

