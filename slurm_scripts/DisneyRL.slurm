#!/bin/bash
#SBATCH --partition oermannlab
#SBATCH --nodes 1
#SBATCH --nodelist a100-8003
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --mem 200G
#SBATCH --time 3-00:00:00
#SBATCH --job-name DisneyRL
#SBATCH --output "/gpfs/data/oermannlab/users/hz2212/DS-GA 3001-007/slurm_logs/DisneyRL-%J.log" 

# load modules or conda environments here
# e.g. farnam:
# module load Python/2.7.13-foss-2016b 
# module add anaconda2/gpu/5.2.0

module load anaconda3/cpu/5.3.1
source activate ds3001-007

export PYTHONPATH=""

cd "$HOME/DS-GA 3001-007"

python train_agent.py --algo $1