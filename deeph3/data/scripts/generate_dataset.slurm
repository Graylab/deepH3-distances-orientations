#!/bin/bash -l

#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mail-user=jruffolo@jhu.edu
#SBATCH --mail-type=ALL

# Module loads
module load cuda
module load python/3.6

pip install --user torch torchvision
pip install --user h5py
pip install --user tqdm
pip install --user biopython
pip install --user pandas

date
# Add project to path
export PYTHONPATH="$PYTHONPATH:/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction"

export SCRIPT=/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/cli/generate_h5_antibody_files_cli.py

export PDBDIR=/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/data/antibody_dataset/pdbs_validation
export FASTADIR=/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/data/antibody_dataset/fastas
export H5FILE=/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/data/antibody_dataset/h5s_dist_angle/antibody_validation.h5

time python $SCRIPT $PDBDIR $FASTADIR $H5FILE --overwrite True --get_seq_from_pdb_file True
date

