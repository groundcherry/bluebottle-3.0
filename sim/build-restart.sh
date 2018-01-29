#!/bin/bash
# build-restart.sh

# Input arguments:
# $1 -- $SLURM_NNODES
# $2 -- $SLURM_NTASKS_PER_NODE
# $2 -- $SLURM_NTASKS_PER_NODE (gres)
# $3 -- $SLURM_CPUS_PER_TASK
# $4 -- $SLURM_JOB_NAME
# $5 -- $SLURM_JOB_ID
# $6 -- $SLURM_NTASKS

#
echo "#!/bin/sh" > submit-restart.sh
echo "#" >> submit-restart.sh
echo "#SBATCH --partition=gpu" >> submit-restart.sh
echo "#SBATCH --time=01:00:00" >> submit-restart.sh
echo "#SBATCH --nodes=$1" >> submit-restart.sh
echo "#SBATCH --ntasks-per-node=$2" >> submit-restart.sh
echo "#SBATCH --gres=gpu:$2" >> submit-restart.sh
echo "#SBATCH --cpus-per-task=$3" >> submit-restart.sh
echo "#SBATCH --job-name=$4" >> submit-restart.sh
echo "#SBATCH --dependency=afterok:$5" >> submit-restart.sh
echo "#SBATCH --output=bbottle.out" >> submit-restart.sh
echo "#SBATCH --open-mode=append" >> submit-restart.sh
echo "" >> submit-restart.sh
echo "sleep 1" >> submit-restart.sh


echo "./build-restart.sh \$SLURM_NNODES \$SLURM_NTASKS_PER_NODE \\
  \$SLURM_CPUS_PER_TASK \$SLURM_JOB_NAME \$SLURM_JOB_ID \$SLURM_NTASKS" >> submit-restart.sh


echo "sbatch -n$6 submit-restart.sh" >> submit-restart.sh
echo "mpirun ./bluebottle -r" >> submit-restart.sh
chmod +x submit-restart.sh
