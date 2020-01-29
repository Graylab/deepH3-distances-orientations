#!/bin/bash -l

#SBATCH --job-name=jackhmmer_aln_%A_%a
#SBATCH --partition=parallel
#SBATCH --time=24:00:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=24
#SBATCH --mem=115GB
#SBATCH --array=1-1462
#SBATCH --output /home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data/scripts/outerr/outerr_jackhmmer_aln_test.%A_%a.out  # Name of stdout output file (%A expands to jobId and %a expands to the array index), remember to make outerr directory
#SBATCH --error /home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data/scripts/outerr/outerr_jackhmmer_aln_test.%A_%a.err  # Name of stdout output file (%A expands to jobId and %a expands to the array index), remember to make outerr directory
#SBATCH --mail-user=cguerra3@jhu.edu
#SBATCH --mail-type=ALL

# Module loads

date
# Define vars for directories
export OUT_DIR=/home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data
export FASTA_DIR=/home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/deep-H3-loop-prediction/deeph3/data/flanked_h3_fastas
export HMMER_DIR=/home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/hmmer-3.2.1/src

export UNI_DB=/home-1/cguerra3@jhu.edu/work/cguerra3@jhu.edu/uniparc_active.fasta
export JACKHMMER=$HMMER_DIR/jackhmmer
export HMMBUILD=$HMMER_DIR/hmmbuild

# Define vars for input/output files
export FASTA_NAME=$(ls $FASTA_DIR/*.fasta | sed -n ${SLURM_ARRAY_TASK_ID}p | xargs -I file basename file .fasta)
export JACKHMMR_OUT_FILE=$OUT_DIR/jackhmmer_out/${FASTA_NAME}_jackhmmer_out.txt
export ALN_FILE=$OUT_DIR/alignment_jackhmmer/${FASTA_NAME}_jackhmmer_aln.txt
export PROFILE_OUT_FILE=$OUT_DIR/profiles/${FASTA_NAME}_profile.txt
export FASTA_FILE=$FASTA_DIR/${FASTA_NAME}.fasta

if [ ! -f $JACKHMMR_OUT_FILE ]
then
    time $JACKHMMER -o $JACKHMMR_OUT_FILE -A $ALN_FILE --notextw -N 5 $FASTA_FILE $UNI_DB
    time $HMMBUILD $PROFILE_OUT_FILE $ALN_FILE
fi
date

