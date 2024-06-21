#!/bin/bash
#SBATCH -J GeEval20
#SBATCH -p vermeer
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
python -u piaa_eval.py --n-at=4 --ns-trial 1 --ns-robo 20 --es-robo 1 2 4 8 12 16 20 --versions 3 --eval-version=3 --n-fitness=7 --save-movie=1 --env-name=NoPinkLine > get_eval_20robo_.out
deactivate