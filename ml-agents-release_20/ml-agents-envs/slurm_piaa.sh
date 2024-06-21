#!/bin/bash
#SBATCH -J 20robo
#SBATCH -p vermeer
#SBATCH -N 4
#SBATCH -n 64
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 64 --oversubscribe python train_mpi_evojax.py --max-iter=500 --base=piaa --ns-robo 20 --n-at=4 --version=3 --n-fitness=7 --ns-trial 3 --load-model=log/at4/1robo/v3/trial_1/Iter_500.npz
deactivate

