#!/bin/bash
#SBATCH -J 9_4-8_1-3
#SBATCH -p vermeer
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --max-iter=1000 --base=piaa --ns-robo 4 8 --n-at=9 --version=1 --n-fitness=7 --ns-attempt 1 2 3
deactivate

