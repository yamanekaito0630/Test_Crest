#!/bin/bash
#SBATCH -J 20
#SBATCH -p matisse
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --max-iter=1000 --base=piaa --ns-robo 20 --n-at=4 --version=3 --n-fitness=7 --ns-trial 1 2 3 --t=2
deactivate

