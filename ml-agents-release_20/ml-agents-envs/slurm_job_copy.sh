#!/bin/bash
#SBATCH -J pi_unity
#SBATCH -p matisse
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o stdout.%J.log
#SBATCH -e stderr.%J.log

source ../../env/bin/activate
python test_env.py
deactivate
