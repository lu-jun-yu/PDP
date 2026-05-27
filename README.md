# PDP-Bench

![Two-stage view of PDP](assets/Two_Stage.png)

## 简介 / Overview

刑事法律判决预测（Legal Judgment Prediction, LJP）通常以已经进入审判程序的案件为对象，预测罪名、法条和刑期。但在审判之前，检察机关已经通过审查起诉决定哪些案件进入审判、哪些案件因证据不足、依法不负刑事责任或有罪但可免予处罚而被不起诉分流。因此，审判阶段 LJP 天然看不到三类关键刑事责任状态。

Criminal Legal Judgment Prediction (LJP) usually operates on cases that have already entered trial, predicting charges, law articles, and prison terms. Before trial, however, the procuratorate has already decided which cases should be prosecuted and which should be diverted through non-prosecution decisions due to insufficient evidence, legally excluded liability, or guilt without punishment. Trial-stage LJP therefore misses three core criminal-liability outcomes.

为补上这一空白，本项目提出 **Prosecution Decision Prediction (PDP, 公诉决定预测)**：将评测锚点前移到审查起诉阶段，要求模型根据犯罪嫌疑人信息、案件程序信息和案件事实，在四类公诉决定之间作出预测。PDP 不只是更早阶段的分类任务，它同时考察模型的证据评价、法律涵摄和价值裁量能力。

To fill this gap, this project proposes **Prosecution Decision Prediction (PDP)**, moving the evaluation anchor to prosecutorial review. Given suspect information, procedural information, and factual information, a model predicts one of four prosecutorial decisions. PDP is not merely an earlier-stage classification task: it evaluates evidence evaluation, legal subsumption, and value-based discretion.

四类标签 / Four labels:

- `P`: 起诉 / Prosecution
- `DNP`: 相对不起诉 / Discretionary Non-Prosecution
- `SNP`: 法定不起诉 / Statutory Non-Prosecution
- `IENP`: 存疑不起诉 / Insufficient-Evidence Non-Prosecution

PDP-Bench 当前包含 **4,630** 条公开中文检察文书样本，时间跨度为 **2014 年 1 月至 2026 年 3 月**，覆盖中国大陆 **31 个省级行政区**。每条样本保留公开来源链接，并包含结构化字段、适用法条、公诉决定和原始审查意见等信息。

PDP-Bench contains **4,630** publicly released Chinese prosecutorial decisions from **January 2014 to March 2026**, covering all **31 provincial-level administrative regions** of mainland China. Each sample retains its public source URL and provides structured fields, cited law articles, the prosecution decision, and original prosecutorial reasoning.

![PDP-Bench overview](assets/pdp_bench_overview.png)

## 研究主线 / Research Questions

本项目围绕三个问题展开实验：

This project studies three empirical questions:

1. PDP 是否对当前 SOTA LLM 构成系统性挑战？  
   Does PDP challenge current SOTA LLMs?

2. test-time scaling、法律领域专门化和提示增强能否解决 PDP？  
   Can test-time scaling, legal-domain specialization, or prompt-side knowledge augmentation solve PDP?

3. 类别增强的 RLVR / DAPO 训练能否提升模型在少数类责任边界上的判别能力？  
   Can class-augmented RLVR / DAPO improve discrimination on minority prosecution-decision boundaries?

实验报告 `Macro-F1`、`Micro-F1`、类别级 F1，并同时评估两种粒度：`PDP L1` 为起诉 / 不起诉二分类，`PDP L2` 为四分类。

Experiments report `Macro-F1`, `Micro-F1`, and class-level F1 at two granularities: `PDP L1` for binary NP/P classification and `PDP L2` for the four-way PDP setting.

## 实验结果 / Experimental Results

### LJP-PDP 能力差距 / LJP-PDP Gap

![LJP-PDP metric correlation](assets/ljp_pdp_l2_metric_correlation.png)

PDP-Bench 暴露出传统审判阶段 LJP 难以覆盖的责任边界，尤其体现在 `SNP` 与 `DNP` 等类别上。  
PDP-Bench exposes liability boundaries that trial-stage LJP does not fully cover, especially on `SNP` and `DNP`.

### RQ3 类别占比干预 / RQ3 Class-Prior Intervention

![RQ3 class-prior intervention](assets/rq3_class_prior_intervention.png)

提高目标类别训练占比主要改变预测偏好，并不稳定转化为可泛化的类别级 F1 提升。  
Increasing the target-class training proportion mainly shifts prediction preferences and does not consistently translate into generalizable class-level F1 gains.

### DAPO 训练奖励曲线 / DAPO Reward Curves

![RQ3 reward curves](assets/rq3_reward_curves.png)

训练奖励曲线用于观察不同类别干预组在 DAPO 训练过程中的优化动态。  
The reward curves show optimization dynamics across different class-prior intervention groups during DAPO training.

## 仓库结构 / Repository Layout

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
`-- README.md
```

本地数据、模型权重、实验输出和论文中间产物请放在本机约定目录中，不要提交到仓库。

Keep local datasets, model weights, experiment outputs, and paper build artifacts out of version control.

## 环境准备 / Setup

建议使用 Linux / WSL / CUDA 环境运行训练和 vLLM 评测。OpenRouter API 评测可在普通 Python 环境中运行。

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

## 数据约定 / Data Layout

评测脚本默认读取本地 HuggingFace `DatasetDict` 目录：

Evaluation scripts expect local HuggingFace `DatasetDict` directories:

- `data/pdp4k`: PDP-Bench test data, usually with `test` and `test_rq2` splits
- `data/pdp2k_rq3`: RQ3 training data with class-prior intervention splits

常见 RQ3 split / Common RQ3 splits:

- `natural`
- `balanced`
- `P-40`, `P-55`
- `DNP-40`, `DNP-55`
- `SNP-40`, `SNP-55`
- `IENP-40`, `IENP-55`

## 评测 / Evaluation

### vLLM

```bash
python eval/evaluate_vllm.py \
  --model-path models/Qwen3-8B \
  --data-path data/pdp4k \
  --split test \
  --output-dir results/vllm/Qwen3-8B
```

常用参数 / Common options:

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

### 指标重算 / Recompute Metrics

```bash
python eval/metrics.py results/RQ1
```

The evaluation output usually includes `details_*.json` and `metrics.md`.

## DAPO 训练 / DAPO Training

快速调试 / Smoke test:

```bash
python train/dapo.py \
  --model-path models/Qwen3-8B \
  --data-path data/pdp2k_rq3 \
  --split balanced \
  --output-dir results/dapo_smoke \
  --max-samples 32
```

多 GPU / DeepSpeed:

```bash
bash scripts/train_RQ3_dapo.sh
```

指定训练组 / Selected groups:

```bash
bash scripts/train_RQ3_dapo.sh DNP-40 DNP-55
```

常用环境变量 / Common environment variables:

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

## 提示词 / Prompts

`prompt_template.py` 提供三种提示词变体：

`prompt_template.py` provides three prompt variants:

- `original`: 基础任务指令 / baseline instruction
- `definitions`: 加入四类决定定义 / adds definitions of the four decisions
- `one_shot`: 加入一组示例 / adds one input-output example

评测与训练脚本通常支持 `--prompt-variant`。

Most evaluation and training scripts support `--prompt-variant`.

## 图表 / Figures

```bash
python visual/plot_pdp_bench_overview.py
python visual/plot_ljp_pdp_l2_class_correlation.py
python visual/plot_rq3_class_prior_intervention.py
python visual/plot_rq3_reward_curves.py
```
