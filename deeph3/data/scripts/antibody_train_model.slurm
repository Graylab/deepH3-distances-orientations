#!/bin/bash -l

#SBATCH --job-name=train_model_%a
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=0
#SBATCH --output /home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/data/scripts/outerr/outerr_train_model.%A_%a.out  # Name of stdout output file (%A expands to jobId and %a expands to the array index), remember to make outerr directory
#SBATCH --error /home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/data/scripts/outerr/outerr_train_model.%A_%a.err  # Name of stdout output file (%A expands to jobId and %a expands to the array index), remember to make outerr directory
#SBATCH --mail-user=jruffolo@jhu.edu
#SBATCH --mail-type=ALL

# Module loads
module load cuda/9.0
module load python/3.6

pip install --user torch==1.1.0 torchvision==0.3.0
pip install --user h5py
pip install --user matplotlib
pip install --user adabound
pip install --user tqdm
pip install --user biopython
pip install --user future
pip install --user tensorboard

date
# Add project to path
export PYTHONPATH="$PYTHONPATH:/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction"

export SCRIPT=/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/cli/antibody_trainer_cli.py
export DATA_DIR=/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/data/antibody_dataset/h5s2
export MODELS_DIR=/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/models/sgd_opt

mkdir $MODELS_DIR

time python $SCRIPT $DATA_DIR --num_blocks1D 3 --num_blocks2D 25 --batch_size 4 --models_dir $MODELS_DIR --epochs 20 --save_every 0 --lr 0.01 --momentum 0.7
date
