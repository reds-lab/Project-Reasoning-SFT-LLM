#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=2         # Request 2 CPU cores
#SBATCH --time=12:00:00             # Set a 12-hour time limit
#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU

# Evaluate a model on all datasets under ./data (one after another).
# Usage: sbatch eval.sh
#        MODEL=/path/to/your/model sbatch eval.sh
# To evaluate a single dataset, use eval_single.sh instead.

# eval_single.sh loads prompts/data relative to the working directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

for DATA in $(ls data); do
    bash eval_single.sh "$DATA"
done
