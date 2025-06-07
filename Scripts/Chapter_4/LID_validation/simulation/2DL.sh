#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --array=37-45
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00

source /mnt/pool/6/vvkulagin/FESTIM/miniconda3/bin/activate
conda activate festim-env2
export DIJITSO_CACHE_DIR=./cache

#export MPLCONFIGDIR=./cache

duration=1ms

config=./config_${duration}.txt

E=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
a=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)


name=data_${duration}_E${E}J_a${a}_${SLURM_JOB_ID}
#name=flux_1ms_E${E}_${SLURM_JOB_ID}
mkdir ${name}

cp -rf properties.py ./${name}
cp -rf 2DL.py ./${name}
cp -rf properties.py ./${name}
cp -rf ./mesh ./${name}/mesh
cd ${name}

mpirun -np 8 python3 2DL.py ${E} ${duration} ${a} 

rm -rf ./cache