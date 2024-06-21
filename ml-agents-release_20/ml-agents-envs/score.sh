#!/bin/bash
#SBATCH -J GetScore
#SBATCH -p matisse
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
python -u piaa_score.py --n-at=4 --version=3 --eval-version=3 --env-name=default --trials 3 --ns-robo 20 --es-robo 4 8 12 16 20 --n-fitness=7 > score.out
deactivate

