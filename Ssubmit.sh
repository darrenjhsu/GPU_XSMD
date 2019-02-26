#!/bin/bash

#SBATCH --job-name CUDA_test
#SBATCH --partition=KEPLER
#SBATCH --qos normal
#SBATCH --nodes 1
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --time 05:20:00
#SBATCH --output CUDA_test.out
#SBATCH --mem=24000

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/djh992/lib/gsl/lib
module load cuda/9.1.85
date +%s%3N
date
DSET=1f6s_2
#for i in {1..100}; do
#srun --gres=gpu:1 tclsh test.tcl
#done
#srun --gres=gpu:1 nvprof --analysis-metrics -o speedtest25.txt ./a.out 
#srun --gres=gpu:1 nvprof ./structure_calc
#srun --gres=gpu:1 ./structure_calc
#srun --gres=gpu:1 cuda-memcheck ./structure_calc 
#srun --gres=gpu:1 cuda-memcheck ./fit_initial 
#srun --gres=gpu:1 ./a.out
#srun --gres=gpu:1 cuda-memcheck ./a.out
#cuda-memcheck ./a.out
#srun --gres=gpu:1 nvprof ./a.out
#nvprof --analysis-metrics -o speedtest_bin12.txt bin/fit_initial.out
nvprof ./bin/$DSET/fit_traj_initial.out
#nvprof ./bin/$DSET/fit_initial.out
#cd bin/2lao
#nvprof ./speedtest.out
cp scat_param.{cu,hh} data/$DSET
date +%s%3N
date
