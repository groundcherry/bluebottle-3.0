#!/bin/sh
# submit.sh
#
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --job-name=example
#SBATCH --output=bbottle.out

#./build-restart.sh $SLURM_NNODES $SLURM_NTASKS_PER_NODE $SLURM_CPUS_PER_TASK \
#                    $SLURM_JOB_NAME $SLURM_JOB_ID $SLURM_NTASKS
#sleep 1
#sbatch -n$SLURM_NTASKS submit-restart.sh

mpirun --preload-files input/flow.config    \
       --preload-files input/decomp.config  \
       --preload-files input/record.config  \
       --preload-files ./bluebottle         \
       ./bluebottle

# NOTES
# -- gres=gpu:n gives n gpus per node
# -- sbatch -nN runs N separate MPI tasks (N total gpus)
 
# Usage:
# marcc: sbatch -n2 submit.sh
#         (inside submit.sh: mpirun ./bluebottle)
# lucan: salloc -n2 --gres=gpu:2 mpirun ./bluebottle
# Use "#SBATCH --open-mode=append" for appending output on repeated jobs.

# TODO for build-restart:
# -- add ??$SLURM_TIME or something so time doesn't have to be changed
# -- add partition as well
# -- add output file name

