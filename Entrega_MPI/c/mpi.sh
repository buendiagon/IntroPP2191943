#!/bin/bash
#SBATCH -o output.log
#SBATCH -J heat_distribution
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

module load devtools/mpi/openmpi/3.1.4

make clean
make

mpirun -np 4 ./heat_mpi bottle.dat 1000