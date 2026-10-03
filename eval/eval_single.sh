#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=2         # Request 2 CPU cores
#SBATCH --time=4:00:00              # Set a 4-hour time limit
#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU

# Evaluate a model on a single dataset.
# Usage: sbatch eval_single.sh <dataset>        e.g. sbatch eval_single.sh amc
#        MODEL=/path/to/your/model sbatch eval_single.sh amc
#        CONDA_ENV=myenv GPU_MEM_UTIL=0.5 bash eval_single.sh amc     (without SLURM, shared GPU)
# Available datasets: the folder names under ./data

DATA=$1

if command -v module &> /dev/null; then
    module load Miniconda3
    module load CUDA/12.6.0
fi

source activate ${CONDA_ENV:-myenv}

# eval.py loads prompts/data relative to the working directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [ -z "$DATA" ] || [ ! -f "data/$DATA/test.jsonl" ]; then
    echo "Usage: sbatch eval_single.sh <dataset>"
    echo "Available datasets: $(ls data | tr '\n' ' ')"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}   # or /path/to/your/model
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.96}          # lower (e.g. 0.5) if the GPU is shared
mkdir -p "$OUTPUT_DIR"

# Leaderboard setup: temperature 0.6, top-p 0.95, 8 samples, 32768 max tokens
python eval.py \
--model_name_or_path "$MODEL" \
--data_name "$DATA" \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--top_p 0.95 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 8 \
--k 1 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--surround_with_messages \
--output_dir "$OUTPUT_DIR" \
--completions_save_dir "$OUTPUT_DIR/completions" \
--gpu_memory_utilization "$GPU_MEM_UTIL" \
2>&1 | tee "$OUTPUT_DIR/log_${DATA}.txt"
