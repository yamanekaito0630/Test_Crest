#!/bin/bash
#SBATCH -J 12_1
#SBATCH -p matisse
#SBATCH -N 3
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --max-iter=1000 --base=piaa --ns-robo 12 --n-at=4 --version=1 --n-fitness=4 --ns-attempt 1 --is-resume=1 --from-iter=701 --init-best=36.11999985575676
deactivate

