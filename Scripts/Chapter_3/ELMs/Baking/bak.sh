#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --array=1-66
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=1 
export DIJITSO_CACHE_DIR=./cache

config=./config.txt

f_ELM=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
q_stat=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
E_ELM=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
Edt=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
eta_tr=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
filename=ELM_${SLURM_ARRAY_JOB_ID}_${f_ELM}Hz_${q_stat}MW_${Edt}eV_${eta_tr}

mkdir ${filename}

cp -rf exposure.py ${filename}
cp -rf baking.py ${filename}

cd ./${filename}

mkdir ./results

mpirun -np 1 python3 exposure.py ${q_stat} ${f_ELM} ${E_ELM} ${eta_tr} ${Edt}
mpirun -np 1 python3 baking.py ${eta_tr} ${Edt}

rm -rf ./cache