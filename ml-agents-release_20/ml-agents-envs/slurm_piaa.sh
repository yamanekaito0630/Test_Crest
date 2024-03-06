#!/bin/bash
#SBATCH -J 16_2-3
#SBATCH -p vermeer
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --max-iter=1000 --base=piaa --ns-robo 16 --n-at=4 --version=1 --n-fitness=6 --ns-trial 2 3
deactivate

