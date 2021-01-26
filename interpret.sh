#!/bin/bash
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=24:ompthreads=4:mem=180gb
#PBS -j oe
cd $PBS_O_WORKDIR
module load lang/python/anaconda/3.7-2019.03
source activate tundra
python interpret.py