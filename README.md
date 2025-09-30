# Assignment-Reasoning-SFT-LLM
SFT Training LLM for Improved Reasoning

# SFT-LLM Assignment: Fine-tuning Qwen2.5-3B-Instruct on AceReason-1.1-SFT

This assignment will guide you through the process of supervised fine-tuning (SFT) a large language model, [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), to improve reasoning. You will first benchmark the base model's performance, then fine-tune it on the AceReason-1.1-SFT subset, and finally evaluate the fine-tuned model to measure the improvement on mathematical and general reasoning benchmarks.

**DEADLINE 0 (Self Team Assignment - Please Check Canvas!): Wednesday Sep 24, 2025 [11:59ET] ** 
**DEADLINE: Friday Oct 10, 2025 [11:59ET] ** 



Late submissions will not be accepted.
If you have questions or feedback, please submit an issue or contact Hoang [just@vt.edu].


OUTCOMES:

- Learning to setup training and evaluation environment.
- Efficient SFT training large language models (LLMs) on reasoning data.
- Robustly evaluating LLMs on diverse benchmarks.
- Studying the impact of training data on the model performance.

Helpful Links:

- Model: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- Dataset: [AceReason-1.1-SFT SUBSET 100K](https://huggingface.co/datasets/redsgnaoh/acereason11_100k)
- LLaMA-Factory: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Eval: [lighteval](https://github.com/huggingface/lighteval)
- ARC On Demand: [Instructions](https://www.docs.arc.vt.edu/get_started.html)


## 0. Setting Up Your Computing Environment

Before we dive into the code, let's set up the high-performance computing environment you will use for this project. We will be using Virginia Tech's **Advanced Research Computing (ARC)** services, which provide access to the powerful GPUs required for training large language models.

First, if you have not already done so, please follow the official documentation to **[get started with an ARC account](https://www.docs.arc.vt.edu/get_started.html)**.

On ARC, you have two primary ways to run your code, each suited for different stages of your workflow:

1.  **Interactive Jobs:** Ideal for development, debugging, and running short experiments. You can use familiar tools like Jupyter Notebooks/Lab.
2.  **Batch Jobs (SLURM):** Necessary for long-running processes, such as the final model training, which may take several hours.

Let's cover how to use both.

### 0.1. Interactive Development with Jupyter

For the development phase of your project, an interactive Jupyter session is the most convenient option. This allows you to write and test code in real-time.

You can launch a new Jupyter session directly from the ARC OnDemand dashboard: **[Launch Jupyter Session](https://ood.arc.vt.edu/pun/sys/dashboard/batch_connect/sys/bc_vt_jupyter/session_contexts/new)**.

When configuring your session, be sure to request the necessary resources (e.g., a GPU and sufficient memory) to run your code.

### 0.2. Submitting a Batch Training Job with SLURM

Once your code is finalized and ready for a full training run, you should submit it as a batch job using the SLURM scheduler. This allows your job to run for an extended period without requiring your active attention.

**Step 1: Create a SLURM Script**

First, create a shell script file named `run_training.sh`. This file will contain both the configuration for the SLURM scheduler and the command to execute your training script.

```bash
#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=2         # Request 2 CPU cores
#SBATCH --time=1:00:00              # Set a 1-hour time limit
#SBATCH --partition=h200_normal_q    # Specify the GPU partition
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU

module load Miniconda3
module load CUDA/12.6.0

#-- Your Code Execution --#
# Ensure you are in the correct conda environment if necessary
source activate your_env_name

# Command to start the fine-tuning job
llamafactory-cli train yamls/your_sft_config.yaml

# If running a custom python script instead:
# python my_training_script.py
```

Please check for the available GPU partitions: [here](https://dashboard.arc.vt.edu/d/feavoh81ioglca/arc-cluster-job-resources-availability?orgId=1&from=now-24h&to=now&timezone=browser&var-datasource=$__all&var-partition=$__all&var-cpus=1&var-gpus=1&var-memory=8&refresh=1m).

**Step 2: Submit the Job**

To submit your job, open a terminal session on ARC: **[ARC Shell Access](https://ood.arc.vt.edu/pun/sys/shell/ssh/tinkercliffs2.arc.vt.edu)**.

Navigate to the directory containing your `run_training.sh` script and use the `sbatch` command to add it to the queue:

```bash
sbatch run_training.sh
```

The system will respond with a job ID. You can monitor the status of your job using commands like `squeue -u <YourPID>` or check the output files once it completes.

BUT we will not run the training at this stage :)  We will setup the environment first and follow up with the evaluation. 

---

Now that you know how to interact with the ARC environment, we can proceed with setting up the project, training the model, and evaluating its performance.


## 1. Environment Setup

First, before running any code, you need to set up the environments for training and evaluation. It's best to clone all the necessary repositories from the start.

**0.1 Enter ARC Terminal**

We will create our environment on the ARC system. 

First, enter the terminal either through: [ARC Shell](https://ood.arc.vt.edu/pun/sys/shell/ssh/tinkercliffs2.arc.vt.edu) or through ssh `ssh username@tinkercliffs2.arc.vt.edu`. 
We then load necessary modules as follows:

```bash
module load Miniconda3
module load CUDA/12.6.0
```


**1.0 Setup Huggingface Token**

Create an account and token (read and write access) on Huggingface: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens).

Then go here and access the GPQA dataset repository: [https://huggingface.co/datasets/Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa).


**1.1. [LLaMA-Factory](LLaMA-Factory) for Training**

```bash
git depth -1 clone https://github.com/hiyouga/LLaMA-Factory.git 
cd LLaMA-Factory
conda create -n myenv python=3.10 # if you are familiar with uv, feel free to use uv instead
source activate myenv
pip install -r requirements.txt
pip install -e ".[torch,metrics]" --no-build-isolation
```

**1.2. [lighteval](https://github.com/huggingface/lighteval) Evaluation**


```bash
git clone https://github.com/huggingface/lighteval.git
cd lighteval/
pip install lighteval[vllm,extended_tasks,math,dev] 
pip install -e .
```

Use `conda list` to check if your packages are installed.

**1.3 (Optional) Flash Attention 2**

We can install Flash Attention through pip if not already installed:

```bash
pip install flash-attn --no-build-isolation
```

**Remember to activate the `myenv` conda environment for all steps.**

**1.4. Downloading the Subset of AceReason-1.1-SFT to local directory**

We will download subset of [AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT) of size 100K.
This will be the dataset we mainly use to train our models.

```

from huggingface_hub import hf_hub_download
# Login using e.g. `huggingface-cli login` to access this dataset

hf_hub_download(repo_id="redsgnaoh/acereason11_100k", repo_type="dataset", filename="data/acereason11_100k.json", local_dir="/path/to/yourdata")

```

The `acereason11_100k.json` file is in a 'training-ready' format. (More information below.)




## 2. Base Model Evaluation

Before fine-tuning, it is crucial to establish a performance baseline. You will evaluate the original `Qwen/Qwen2.5-3B-Instruct` model on the same benchmarks you will use for the fine-tuned version. 


### 2.1. Mathematical Reasoning with Lighteval

Here, we will evaluate the base model's reasoning capability on math and science benchmarks using `lighteval` package.
For this, we include the popular reasoning benchmarks, including [AIME2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), [AIME2025](https://huggingface.co/datasets/opencompass/AIME2025), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) for math, [GPQA-Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) for science, and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) for code generation.

You can either create bash file or run directly:

```bash
#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=2         # Request 2 CPU cores
#SBATCH --time=1:00:00              # Set a 1-hour time limit
#SBATCH --partition=h200_normal_q    # Specify the GPU partition
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU

module load Miniconda3
module load CUDA/12.6.0

source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=2
MODEL=Qwen/Qwen2.5-3B-Instruct
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=your/output/dir

lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0,extended|lcb:codegeneration|0|0" \
    --save-details \
--output-dir $OUTPUT_DIR
```

### 2.2. General Benchmarks with Lighteval

Here, we will run the evaluation on the [MMLU-Redux-2](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0) benchmark to measure the general performance of the model using `lighteval`.
Please read [https://huggingface.co/docs/lighteval/main/en/quicktour](https://huggingface.co/docs/lighteval/main/en/quicktour) for more information.

You can either create bash file or run directly:

```bash
#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=2         # Request 2 CPU cores
#SBATCH --time=1:00:00              # Set a 1-hour time limit
#SBATCH --partition=h200_normal_q    # Specify the GPU partition
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU

module load Miniconda3
module load CUDA/12.6.0

source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=2
MODEL=Qwen/Qwen2.5-3B-Instruct
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=your/output/dir

lighteval vllm $MODEL_ARGS "lighteval|mmlu_redux_2|0|0" \
--output-dir $OUTPUT_DIR

```
Record the scores from these evaluations. They will be your baseline for comparison.

## 3. Dataset Creation from AceReason-1.1-SFT



Here, we create a training dataset to train our model. We randomly select a subset of 15K data from AceReason-1.1-SFT 100K subset (`acereason11_100k.json`). Remember, we downloaded the file in step 1.4.
Then, we will structure the selected data in the format suitable for LLaMA-Factory, as shown here: [https://raw.githubusercontent.com/GAIR-NLP/LIMO/refs/heads/main/train/data/limo.json](https://raw.githubusercontent.com/GAIR-NLP/LIMO/refs/heads/main/train/data/limo.json). 

```
[
    {
        "instruction": "Find the last three digits of the product of the positive roots of $\\sqrt{1995}x^{\\log_{1995}x}=x^2.$",
        "input": "",
        "output": "Okay, so I need to find the last three digits of the product of the positive roots of the equation √(1995) * x^(log_{1995} x) = x². Hmm, let's see. First, let me try to understand [...] Therefore, the final answer is \\boxed{025}.\n\n**Final Answer**\n\\boxed{025}",
        "system": "Please reason step by step, and put your final answer within \\boxed{}."
    },
...
]
```

Please save the given 15K subset `yourdata.json` file in the [LLaMA-Factory/data](LLaMA-Factory/data) folder.
Lastly, please update the [LLaMA-Factory/data/dataset_info.json](LLaMA-Factory/data/dataset_info.json) file, which manages all datasets for training, by adding the entry of your data file as follows:

```
[
  "yourdata": {
    "file_name": "yourdata.json"
  },
...
]
```


## 4. Supervised Fine-Tuning (SFT) with [LLaMA-Factory](LLaMA-Factory)

Now, you are ready to fine-tune the model on a subset of the AceReason-1.1-SFT dataset to improve the model's math and code reasoning.

Navigate to your `LLaMA-Factory` directory.
```bash
cd LLaMA-Factory
```
Create a YAML configuration file named `sft_config.yaml` to specify the training parameters.

```yaml
# sft_config.yaml
### model
model_name_or_path: Qwen/Qwen2.5-3B-Instruct

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_offload_config.json //Edit if needed
flash_attn: fa2

### dataset
dataset: yourdata  // Edit if needed
cutoff_len: 16384
overwrite_cache: true
preprocessing_num_workers: 16
template: qwen

### output
output_dir: saves/qwen25_3b_instruct/your_name      //Edit accordingly
logging_steps: 1
# save_strategy: epoch
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1      //Edit if needed
gradient_accumulation_steps: 8      //Edit if needed
learning_rate: 5.0e-6               //Edit if needed
num_train_epochs: 10                //Edit if needed
lr_scheduler_type: cosine
warmup_ratio: 0.0                   //Edit if needed
bf16: true
ddp_timeout: 180000000

report_to: wandb
```

To start the training, run the following command in your terminal:

```bash
llamafactory-cli train sft_config.yaml
```

This command will download the model and dataset (if not cached) and save the trained model to the `saves/qwen25_3b_instruct/your_name` directory.



## 6. Fine-Tuned Model Evaluation

After fine-tuning, you will evaluate your new model and compare its performance to the baseline.
Similarly, we will use `lighteval` here:

```bash
#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=2         # Request 2 CPU cores
#SBATCH --time=1:00:00              # Set a 1-hour time limit
#SBATCH --partition=h200_normal_q    # Specify the GPU partition
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU

module load Miniconda3
module load CUDA/12.6.0

source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=2
MODEL=/path/to/your/model
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=your/output/dir

lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0,extended|lcb:codegeneration|0|0,lighteval|mmlu_redux_2|0|0" \
    --save-details \
--output-dir $OUTPUT_DIR
```

Compare these new scores with your baseline to see the impact of fine-tuning.

## 7. Advanced Data Selection Strategy

Now, we proceed to the main task of our project: Choosing Effective Training Data.

Simply selecting a random subset of data is a good starting point, but more advanced data selection strategies can often lead to better performance [[1](https://arxiv.org/pdf/2506.13284), [2](https://arxiv.org/pdf/2506.04178), [3](https://arxiv.org/pdf/2503.18892), [4](https://arxiv.org/pdf/2502.03387)]. The quality of the data used for fine-tuning is crucial.
For this assignment, you will investigate data selection techniques to curate your 15K sample subset from the AceReason-100K dataset. Implementing a smarter data selection strategy could significantly improve your model's final benchmark scores. Here are some approaches to inspire your method:

1. Difficulty-Based Selection

The goal is to select problems to teach the model new reasoning patterns based on difficulty. A simple proxy for difficulty is the the model's correctness, length, and complexity of the solution.

Concept: Shorter solutions may represent problems that are too easy, while extremely long solutions might be overly complex or verbose. The "sweet spot" is often in the middle-to-high range of complexity.

Implementation Idea:

+ Load the AceReason-1.1-SFT subset dataset.

+ For each sample, measure the difficulty of a sample by generating multiple responses and measuring accuracy.

+ Analyze the distribution of this metric.

+ Select the points accordingly.


2. Diversity-Based Selection

You can also consider the coverage of the prompts selected in your dataset.

Ensuring the training data covers a wide range of topics and problem types is key to building a robust and generalizable model.

Concept: If the dataset contains many similar problems, randomly sampling might lead to an over-representation of some problem types and an under-representation of others. Diversity-based selection aims to create a more comprehensive subset.

Implementation Idea:

+ Generate sentence embeddings for all the prompts (user questions) in the dataset using a library like sentence-transformers.

+ Use a clustering algorithm like K-Means to group the prompts into k clusters based on their embeddings. Prompts for similar problems will naturally fall into the same clusters.

+ To build your 15k dataset, sample 15000 / k examples from each cluster. This ensures that your final training data includes a balanced representation of all the different types of problems identified by the clustering algorithm.

+ You can feel free to also add data points from different datasets (apart from AceReason) to your subset.



3. Another approach is presented in **LIMOPro (Large-scale Instruction-following Model based on Prompt-response Optimization)**. This method aims to prune and select data points suitable for a given model. You can explore their methodology and implementation here:

*   **LIMOPro GitHub Repository:** [https://github.com/GAIR-NLP/LIMOPro](https://github.com/GAIR-NLP/LIMOPro)


Main Goal: Can you beat the random baseline? 😉


## 8. Submission

Upon completion of the assignment, you will need to submit the following:

*   A link to your GitHub repository containing all necessary files and your final README.md.
*   Push your models to Huggingface: [https://huggingface.co/docs/hub/en/models-uploading](https://huggingface.co/docs/hub/en/models-uploading).
*   A report summarizing your findings. This report must include:
    *   The baseline performance of the original Qwen2.5-3B-Instruct model.
    *   The performance results of your fine-tuned model.
    *   A comparison and analysis of the results.
    *   The hyperparameters you used for your final model.
    *   A description of the data selection strategy you used.

This assignment will provide you with hands-on experience in fine-tuning large language models and rigorously evaluating their performance. 

## 9. LiveBenchmark

We will post a live benchmark of submitted models.
The TOP 5 teams will get a bonus of 10 points!


Good luck!
