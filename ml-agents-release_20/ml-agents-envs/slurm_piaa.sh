#!/bin/bash
#SBATCH -J 1robo
#SBATCH -p matisse
#SBATCH -N 2
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --log-dir=log/at8/round_im_1_robo_slurm_at8 --max-iter=500 --base=piaa --n-robo=1 --n-at=8 --version=1 --n-fitness=7
deactivate

