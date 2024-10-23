#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 07-00
#SBATCH -o /scratch/mt5383/MLAutoScheduler/scripts/lqcd/auto-outer-p2.out
#SBATCH -e /scratch/mt5383/MLAutoScheduler/scripts/lqcd/auto-outer-p2.err
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

/scratch/mt5383/MLAutoScheduler/build/bin/AutoSchedulerML /scratch/mt5383/MLAutoScheduler/lqcd-benchmarks/ABconj_outer.mlir
/scratch/mt5383/MLAutoScheduler/build/bin/AutoSchedulerML /scratch/mt5383/MLAutoScheduler/lqcd-benchmarks/ABCD_outer.mlir
/scratch/mt5383/MLAutoScheduler/build/bin/AutoSchedulerML /scratch/mt5383/MLAutoScheduler/lqcd-benchmarks/ABCD_complex_outer.mlir
/scratch/mt5383/MLAutoScheduler/build/bin/AutoSchedulerML /scratch/mt5383/MLAutoScheduler/lqcd-benchmarks/ABCD_ABconj_outer.mlir
/scratch/mt5383/MLAutoScheduler/build/bin/AutoSchedulerML /scratch/mt5383/MLAutoScheduler/lqcd-benchmarks/ABCD_ABconj_nameless_outer.mlir
