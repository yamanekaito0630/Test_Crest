#!/bin/bash
#SBATCH -J pi_unity
#SBATCH -p matisse
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
xvfb-run -s "-screen 0 800x600x24" -s "-screen 1 800x600x24" -s "-screen 2 800x600x24" -s "-screen 3 800x600x24" -s "-screen 4 800x600x24" -s "-screen 5 800x600x24" -s "-screen 6 800x600x24" -s "-screen 7 800x600x24" mpirun --oversubscribe -n 4 python train_mpi.py --population-size=4 --log-dir=log/marine_drones_mpi_slurm
deactivate
