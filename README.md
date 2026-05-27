# PDP-Bench

English | [中文](README.zh-CN.md)

**Paper:** Preprint coming soon

![Two-stage view of PDP](assets/Two_Stage.png)

## Overview

Criminal Legal Judgment Prediction (LJP) usually operates on cases that have already entered trial, predicting charges, law articles, and prison terms. Before trial, however, the procuratorate has already decided which cases should be prosecuted and which should be diverted through non-prosecution decisions due to insufficient evidence, legally excluded liability, or guilt without punishment. Trial-stage LJP therefore misses three core criminal-liability outcomes.

To fill this gap, this project proposes **Prosecution Decision Prediction (PDP)**, moving the evaluation anchor to prosecutorial review. Given suspect information, procedural information, and factual information, a model predicts one of four prosecutorial decisions. PDP is not merely an earlier-stage classification task: it evaluates evidence evaluation, legal subsumption, and value-based discretion.

Four labels:

- `P`: Prosecution
- `DNP`: Discretionary Non-Prosecution
- `SNP`: Statutory Non-Prosecution
- `IENP`: Insufficient-Evidence Non-Prosecution

PDP-Bench contains **4,630** publicly released Chinese prosecutorial decisions from **January 2014 to March 2026**, covering all **31 provincial-level administrative regions** of mainland China. Each sample retains its public source URL and provides structured fields, cited law articles, the prosecution decision, and original prosecutorial reasoning.

![PDP-Bench overview](assets/pdp_bench_overview.png)

## Research Questions

This project studies three empirical questions:

1. Does PDP challenge current SOTA LLMs?
2. Can test-time scaling, legal-domain specialization, or prompt-side knowledge augmentation solve PDP?
3. Can class-augmented RLVR / DAPO improve discrimination on minority prosecution-decision boundaries?

Experiments report `Macro-F1`, `Micro-F1`, and class-level F1 at two granularities: `PDP L1` for binary NP/P classification and `PDP L2` for the four-way PDP setting.

## Experimental Results

### LJP-PDP Gap

![LJP-PDP metric correlation](assets/ljp_pdp_l2_metric_correlation.png)

PDP-Bench exposes liability boundaries that trial-stage LJP does not fully cover, especially on `SNP` and `DNP`.

### RQ3 Class-Prior Intervention

![RQ3 class-prior intervention](assets/rq3_class_prior_intervention.png)

Increasing the target-class training proportion mainly shifts prediction preferences and does not consistently translate into generalizable class-level F1 gains.

### DAPO Reward Curves

![RQ3 reward curves](assets/rq3_reward_curves.png)

The reward curves show optimization dynamics across different class-prior intervention groups during DAPO training.

## Repository Layout

```text
.
|-- assets/           # README figures
|-- configs/          # DeepSpeed configs
|-- eval/             # vLLM / OpenRouter evaluation and metrics
|-- scripts/          # RQ1/RQ2/RQ3 batch scripts
|-- train/            # DAPO training and reward function
|-- visual/           # Figure generation scripts
|-- prompt_template.py
|-- requirements.txt
|-- README.md
`-- README.zh-CN.md
```

Keep local datasets, model weights, experiment outputs, and paper build artifacts out of version control.

## Setup

Linux / WSL / CUDA is recommended for training and vLLM evaluation. OpenRouter-based evaluation can run in a standard Python environment.

```bash
conda create -n pdp python=3.11 -y
conda activate pdp
pip install -r requirements.txt
```

OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

## Data Layout

Evaluation scripts expect local HuggingFace `DatasetDict` directories:

- `data/pdp4k`: PDP-Bench test data, usually with `test` and `test_rq2` splits
- `data/pdp2k_rq3`: RQ3 training data with class-prior intervention splits

Common RQ3 splits:

- `natural`
- `balanced`
- `P-40`, `P-55`
- `DNP-40`, `DNP-55`
- `SNP-40`, `SNP-55`
- `IENP-40`, `IENP-55`

## Evaluation

### vLLM

```bash
python eval/evaluate_vllm.py \
  --model-path models/Qwen3-8B \
  --data-path data/pdp4k \
  --split test \
  --output-dir results/vllm/Qwen3-8B
```

Common options:

- `--prompt-variant original|definitions|one_shot`
- `--num-repeats 8`
- `--max-model-len 16384`
- `--max-tokens 8192`
- `--tensor-parallel-size 1`
- `--batch-size 200`

### OpenRouter

```bash
python eval/evaluate_openrouter.py \
  --model qwen/qwen3.6-max-preview \
  --data-path data/pdp4k \
  --split test \
  --output-dir results/RQ1 \
  --prompt-variant original \
  --reasoning-effort auto
```

Batch scripts:

```powershell
.\scripts\eval_RQ1.ps1
.\scripts\eval_RQ2_1.ps1
```

```bash
bash scripts/eval_openrouter.sh
bash scripts/eval_vllm.sh
```

### Recompute Metrics

```bash
python eval/metrics.py results/RQ1
```

The evaluation output usually includes `details_*.json` and `metrics.md`.

## DAPO Training

Smoke test:

```bash
python train/dapo.py \
  --model-path models/Qwen3-8B \
  --data-path data/pdp2k_rq3 \
  --split balanced \
  --output-dir results/dapo_smoke \
  --max-samples 32
```

Multi-GPU / DeepSpeed:

```bash
bash scripts/train_RQ3_dapo.sh
```

Selected groups:

```bash
bash scripts/train_RQ3_dapo.sh DNP-40 DNP-55
```

Common environment variables:

```bash
MODEL_PATH=models/Qwen3-8B \
DATA_PATH=data/pdp2k_rq3 \
OUTPUT_ROOT=results/RQ3_DAPO \
NUM_GPUS=2 \
VLLM_TP=2 \
BATCH_SIZE=8 \
GRAD_ACCUM=64 \
bash scripts/train_RQ3_dapo.sh balanced
```

Evaluate RQ3 checkpoints:

```bash
bash scripts/eval_RQ3.sh
```

## Prompts

`prompt_template.py` provides three prompt variants:

- `original`: baseline instruction
- `definitions`: adds definitions of the four decisions
- `one_shot`: adds one input-output example

Most evaluation and training scripts support `--prompt-variant`.

## Figures

```bash
python visual/plot_pdp_bench_overview.py
python visual/plot_ljp_pdp_l2_class_correlation.py
python visual/plot_rq3_class_prior_intervention.py
python visual/plot_rq3_reward_curves.py
```
