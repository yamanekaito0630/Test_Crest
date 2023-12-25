#!/bin/bash
#SBATCH -J 12robo
#SBATCH -p vermeer
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --log-dir=log/at8/12robo/v1 --max-iter=500 --base=piaa --n-robo=12 --n-at=8 --version=1 --n-fitness=7
deactivate

