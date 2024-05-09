#!/bin/sh
#SBATCH --time=48:00:00 
#SBATCH --job-name sal_eflstm-mosi
#SBATCH --nodes=2
#SBATCH -c 6 # ask for one core
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-user=amkatay21@earlham.edu

echo "queue/partition is `echo $SLURM_JOB_PARTITION`"
echo "running on `echo $SLURM_JOB_NODELIST`"
echo "work directory is `echo $SLURM_SUBMIT_DIR`"

module unload python/3.9
module load python/3.11
python3 sal_eflstm.py ud_saleflstm_mosei