#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --array=1-105
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

export DIJITSO_CACHE_DIR=./cache
export MPLCONFIGDIR=./cache

name=${SLURM_JOB_ID}
mkdir ${name}

cp -rf sub_functions ./${name}
cp -rf model.py ./${name}
cp -rf parametric_study_Edt_phi.py ./${name}
cd ${name}

mpirun -np 1 python3 parametric_study_Edt_phi.py ${SLURM_ARRAY_TASK_ID}

rm -rf ./cache