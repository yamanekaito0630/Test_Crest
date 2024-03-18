#!/bin/bash
#SBATCH -J 20_2
#SBATCH -p vermeer
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --max-iter=1000 --base=piaa --ns-robo 20 --n-at=4 --version=1 --n-fitness=6 --ns-trial 2 --init-best=15.84849950298667 --is-resume=1 --from-iter=301
deactivate

