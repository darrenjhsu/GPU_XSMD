#!/bin/bash

#SBATCH --job-name CUDA_test
#SBATCH --partition=PASCAL
#SBATCH --qos normal
#SBATCH --nodes 1
#SBATCH --cpus-per-task 24
#SBATCH --gres=gpu:2
#SBATCH --time 00:03:00
#SBATCH --output CUDA_test.out



module load cuda/9.1.85
date +%s%3N
#for i in {1..100}; do
#srun --gres=gpu:1 tclsh test.tcl
#done
#srun --gres=gpu:1 nvprof --analysis-metrics -o speedtest8.txt ./a.out 
srun --gres=gpu:1 nvprof ./a.out 
date +%s%3N
