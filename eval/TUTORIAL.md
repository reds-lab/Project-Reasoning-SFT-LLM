# Evaluation Tutorial

This folder evaluates a model (the base `Qwen/Qwen2.5-3B-Instruct` or your fine-tuned checkpoint) on 8 math and science reasoning benchmarks. It uses vLLM for generation and a SymPy-based grader to check the final `\boxed{...}` answer.

## 1. Folder layout

```
eval/
├── eval.sh              # SLURM job: runs every dataset in data/, one after another
├── eval_single.sh       # SLURM job: runs one dataset (the main entry point)
├── eval.py              # generation + answer extraction + scoring
├── requirements.txt     # Python dependencies
├── data/<dataset>/test.jsonl         # benchmark questions and gold answers
├── prompts/qwen-instruct/<dataset>.py # system prompt / question template per dataset
└── utils/
    ├── data_loader.py         # reads data/<dataset>/<split>.jsonl
    ├── parser.py              # gets the question, gold answer, and the model's \boxed{} answer
    ├── grader.py              # check_is_correct(): numeric + symbolic equivalence
    └── math_normalization.py  # LaTeX/string normalization helpers
```

### Benchmarks

| Dataset (`data/` folder) | # Questions | Answer type |
|---|---|---|
| `aime`          | 30  | integer |
| `amc`           | 40  | number |
| `math`          | 499 | LaTeX expression (MATH-500) |
| `minerva`       | 272 | number / expression |
| `olympiadbench` | 675 | number / expression |
| `cn_math_2024`  | 30  | LaTeX expression |
| `kaoyan`        | 199 | mostly multiple-choice letter (Chinese) |
| `gpqa`          | 198 | multiple-choice letter (A–D) |

## 2. Setup

Create a conda environment and install the dependencies:

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
pip install -r requirements.txt
```

Notes:
- `requirements.txt` pins `vllm<=0.6.1`, `sympy==1.12` and `antlr4-python3-runtime==4.11.1`. The antlr version must match what `latex2sympy2` expects, so don't upgrade it.
- `eval_single.sh` runs `source activate ${CONDA_ENV:-myenv}`. If your environment has another name, set `CONDA_ENV=<name>`.
- On the cluster the script loads `Miniconda3` and `CUDA/12.6.0` automatically when the `module` command exists.

## 3. Running an evaluation

Always launch from the `eval/` folder, because `eval.py` reads `./data` and `./prompts` relative to the working directory. The scripts `cd` there for you (`$SLURM_SUBMIT_DIR`, or the script's own folder).

### One dataset

```bash
cd eval
sbatch eval_single.sh amc                                   # base model, default settings
MODEL=/path/to/your/checkpoint sbatch eval_single.sh amc    # your fine-tuned model
```

Running without an argument, or with a name that doesn't exist, prints the list of available datasets.

### All datasets

```bash
cd eval
MODEL=/path/to/your/checkpoint sbatch eval.sh
```

`eval.sh` loops over every folder in `data/` and calls `eval_single.sh` for each one. It requests 12 hours instead of 4.

### Without SLURM (interactive node or shared GPU)

```bash
CONDA_ENV=myenv GPU_MEM_UTIL=0.5 CUDA_VISIBLE_DEVICES=0 bash eval_single.sh aime
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-3B-Instruct` | HF model id or local checkpoint path |
| `OUTPUT_DIR` | `./outputs` | Where results and logs go |
| `GPU_MEM_UTIL` | `0.96` | Fraction of GPU memory vLLM may use (lower it on a shared GPU) |
| `CONDA_ENV` | `myenv` | Conda environment to activate |
| `CUDA_VISIBLE_DEVICES` | `0` | GPUs to use. With more than one GPU, vLLM uses tensor parallelism across all of them |

To change the SLURM partition, time limit or account, edit the `#SBATCH` lines at the top of the scripts (for example `a100_normal_q` instead of `h200_normal_q`).

## 4. Evaluation settings (leaderboard setup)

`eval_single.sh` calls `eval.py` with these settings. Use the same settings for every model so the scores can be compared:

| Flag | Value | Meaning |
|---|---|---|
| `--prompt_type` | `qwen-instruct` | Load prompts from `prompts/qwen-instruct/` |
| `--surround_with_messages` | on | Wrap the prompt with the model's chat template (system + user) |
| `--temperature` / `--top_p` | `0.6` / `0.95` | Sampling parameters |
| `--n_sampling` | `8` | Responses generated per question |
| `--k` | `1` | Report Pass@1 |
| `--max_tokens` | `32768` | Max generation length |
| `--seed` | `0` | Random seed |
| `--start_idx` / `--end_idx` | `0` / `-1` | Evaluate the whole file. Use a small range (e.g. `0`/`10`) for a quick smoke test |

Every dataset uses the same prompt:

- **System:** `Please reason step by step, and put your final answer within \boxed{}.`
- **User:** the question text, taken from the `question`, `problem`, `Question` or `input` field.

## 5. How scoring works

For each question, `eval.py` does the following:

1. Generates `n_sampling` responses with vLLM.
2. **Extracts the answer** (`utils/parser.py: extract_answer`) from the content of the **last** `\boxed{...}` in the response, matching nested braces. If a response has no `\boxed{}`, its answer is empty and is graded wrong.
3. **Grades it** (`utils/grader.py: check_is_correct`) by normalizing both strings and comparing them with `math_equal`. The comparison accepts exact matches, numeric matches within a tolerance, multiple-choice letters, and symbolic equivalence through SymPy/latex2sympy (with a timeout).
4. The gold answer is the `answer` field of the JSONL record.

At the end, the log prints two numbers:

```
correct cnt / total cnt: 23/40
Acc: 0.5750           # fraction of questions where ANY of the 8 samples is correct (≈ pass@8)
Pass@1: 15.25/40 = 0.3812   # unbiased pass@k estimate averaged over questions; with k=1 this is mean per-sample accuracy
```

**Report `Pass@1`.** `Acc` is an "any of 8 correct" score and is much more optimistic.

### Changing the pass@k metric

Two flags in `eval_single.sh` control the metric:

```bash
--n_sampling 8 \   # n: responses generated per question
--k 1 \            # k: the k in pass@k
```

For each question with `c` correct responses out of `n`, `eval.py` (lines 199–214) uses the unbiased estimator from the Codex paper:

```
pass@k = 1 - C(n-c, k) / C(n, k)     (= 1 when n - c < k,  = 0 when c = 0)
```

The printed `Pass@k` is this value averaged over all questions.

**Example: report pass@4 using 16 samples.** Edit `eval_single.sh`:

```bash
--n_sampling 16 \
--k 4 \
```

Rules to follow:

- **Keep `k <= n_sampling`.** If `k > n`, the code doesn't crash. Instead, every question with at least one correct sample scores 1, so you silently get pass@n.
- **Use `n_sampling > 1`.** With `n_sampling 1`, pass@k is skipped and the script prints `Pass@1 = Acc` (greedy-style accuracy). In that case set `--temperature 0` for real greedy decoding.
- **Use a larger `n` than `k`** for a lower-variance estimate (e.g. n=16 for pass@4).
- **Choose `n_sampling` with a divisor between 2 and 64.** vLLM generates `n_sampling` in rounds of size *f*, where *f* is the largest divisor of `n_sampling` between 2 and 64 (lines 99–104). For example, 8 → 1 round of 8, 128 → 2 rounds of 64. A prime above 64 (e.g. 67) gives 67 rounds of 1, which is very slow.
- **Generation cost grows linearly with `n_sampling`**, so raise the SLURM `--time` when you increase it.
- **Changing only `--k` does not regenerate.** The output filename contains `n_sampling` (`..._k{n_sampling}_...`) but not `k`. If you re-run with a new `--k` and the same `n_sampling`, the script finds the existing file and exits without scoring (see Section 6).

**Computing several k values without regenerating.** The results JSONL stores `answers_correctness` for every sample, so you can compute any `k <= n` offline:

```bash
python -c "
import json, sys
from math import comb
rows = [json.loads(l) for l in open(sys.argv[1])]
n = len(rows[0]['answers_correctness'])
for k in [1, 2, 4, 8]:
    if k > n: break
    s = 0
    for r in rows:
        c = sum(r['answers_correctness'])
        s += 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
    print(f'pass@{k}: {s/len(rows):.4f}')
" outputs/Qwen/Qwen2.5-3B-Instruct/amc/test_qwen-instruct_t0.6_k8_s0_e40.jsonl
```

With n=8, `pass@8` here equals the `Acc` line in the log.

## 6. Outputs

With the default `OUTPUT_DIR=./outputs`:

```
outputs/
├── log_<dataset>.txt                          # full stdout (prompt example, progress, final scores)
├── <model>/<dataset>/test_qwen-instruct_t0.6_k8_s0_e<N>.jsonl
└── completions/<model>/<dataset>/..._gen_round<i>.pkl   # raw vLLM RequestOutput objects (pickled)
```

`<model>` is the last three components of the model path (for example `Qwen/Qwen2.5-3B-Instruct`). Each line of the results JSONL has these fields:

```json
{
  "question": "...",
  "generated_responses": ["...", "..."],   // 8 full responses
  "generated_answers": ["27", "27", ...],  // extracted \boxed{} contents
  "gold_answer": "27.0",
  "is_correct": true,                       // any sample correct
  "answers_correctness": [true, false, ...],
  "id": 0                                   // if present in the data
}
```

You can recompute Pass@1 from the file:

```bash
python -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print(sum(sum(r['answers_correctness'])/len(r['answers_correctness']) for r in rows)/len(rows))
" outputs/Qwen/Qwen2.5-3B-Instruct/amc/test_qwen-instruct_t0.6_k8_s0_e40.jsonl
```

> **Caution:** if the results file already exists, `eval.py` prints `Completely same name file ... exist, skip generation` and exits without scoring. To re-run, delete the file or change `OUTPUT_DIR`.

## 7. Adding a new benchmark

1. Create `data/<name>/test.jsonl`. Each line needs a question field (`question` or `problem`) and an `answer` field.
2. Create `prompts/qwen-instruct/<name>.py` that defines `system_prompt`, `few_shot_prompt` and `question_format` (copy `math.py`). If this file is missing, `eval.py` raises `FileNotFoundError`.
3. Run `sbatch eval_single.sh <name>`. `eval.sh` will also pick it up automatically.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA out-of-memory at vLLM startup | Lower `GPU_MEM_UTIL` (e.g. `0.5`), or check that no other process is using the GPU |
| `FileNotFoundError: ./prompts/...` | You launched from the wrong folder, or the dataset has no prompt file |
| Run finishes instantly with "skip generation" | The results file already exists (see Section 6) |
| Very low scores with a fine-tuned model | Check that the checkpoint includes its tokenizer and chat template, and that the model writes `\boxed{}`. Look at `generated_answers` in the JSONL |
| Job hits the time limit | `max_tokens=32768` × 8 samples is expensive on large sets (`olympiadbench`, `math`). Run datasets as separate `eval_single.sh` jobs |
