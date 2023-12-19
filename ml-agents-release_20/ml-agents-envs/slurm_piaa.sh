#!/bin/bash
#SBATCH -J 20robo
#SBATCH -p matisse
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --log-dir=log/at4/round_im_20_robo_slurm_at4 --max-iter=1000 --base=piaa --n-robo=20 --n-at=4 --version=1 --n-fitness=6
deactivate

