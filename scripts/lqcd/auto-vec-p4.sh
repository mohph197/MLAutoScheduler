#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 02-00
#SBATCH -o /scratch/mt5383/MLAutoScheduler/scripts/lqcd/auto-vec-p4.out
#SBATCH -e /scratch/mt5383/MLAutoScheduler/scripts/lqcd/auto-vec-p4.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mt5383@nyu.edu

#Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

#Activate any environments if required
conda activate main

#Execute the code
export LLVM_PATH=/scratch/mt5383/llvm-project
export SHARED_LIBS=/scratch/mt5383/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/mt5383/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/mt5383/llvm-project/build/lib/libomp.so
export PYTHON=/home/mt5383/.conda/envs/main/bin/python
export AS_VERBOSE=1

/scratch/mt5383/MLAutoScheduler/build/bin/AutoSchedulerML /scratch/mt5383/MLAutoScheduler/lqcd-benchmarks/ABCD_2let_cpx.mlir
/scratch/mt5383/MLAutoScheduler/build/bin/AutoSchedulerML /scratch/mt5383/MLAutoScheduler/lqcd-benchmarks/ABCD_2let_cpx_nameless.mlir
