#!/bin/bash
#SBATCH -J GeData20
#SBATCH -p vermeer
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
python get_data.py --n-at=4 --ns-trial 1 --ns-robo 20 --es-robo 16 --version=3 --reps=1 --env-name=default
deactivate

