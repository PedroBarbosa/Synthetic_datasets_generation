#!/bin/bash
#SBATCH --job-name=RBFOX2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=23G
#SBATCH --gres-flags=enforce-binding
#SBATCH --array=0-4%1
#SBATCH --output=rna_evolution_%j.log

configs=$(readlink -f "$1")

# Activate environment that has dress installed
#source activate dress_experiments

# Load individual configuration files and run dress on each of them using JobArrays
readarray yaml_files < $(readlink -f $configs)
yaml=${yaml_files[$SLURM_ARRAY_TASK_ID]}

CMD="srun dress generate --config $yaml 0_RBFOX2_ES_events.tsv"
echo $CMD
$CMD

#conda deactivate
