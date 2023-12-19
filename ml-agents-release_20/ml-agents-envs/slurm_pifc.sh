#!/bin/bash
#SBATCH -J ir_round_one_robot
#SBATCH -p vermeer
#SBATCH -N 2
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
mpirun -n 32 --oversubscribe python train_mpi_evojax.py --population-size=32 --log-dir=log/round_ir_one_robot_slurm --max-iter=1000 --base=pifc --is-resume=True --init-best=4.957467558910139 --from-iter=151
deactivate

