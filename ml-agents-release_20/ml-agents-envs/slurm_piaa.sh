#!/bin/bash
#SBATCH -J 12_1-3
#SBATCH -p vermeer
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --max-iter=1000 --base=piaa --ns-robo 12 --n-at=9 --version=2 --n-fitness=7 --ns-trial 1 2 3
deactivate

