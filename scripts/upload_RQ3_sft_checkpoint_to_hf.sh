#!/bin/bash
# =============================================================
#  scripts/upload_RQ3_sft_checkpoint_to_hf.sh
#
#  Upload the final RQ3 SFT LoRA adapter directory to Hugging Face Hub.
#
#  Default source:
#    checkpoints/RQ3_SFT_Qwen3-8B_train_seed42_original
#
#  Usage:
#    conda activate pytorch
#    HF_REPO_ID=your-name/PDP-Qwen3-8B-RQ3-SFT-LoRA \
#      bash scripts/upload_RQ3_sft_checkpoint_to_hf.sh
#
#  Common overrides:
#    LOCAL_DIR=checkpoints/.../RQ3_SFT_Qwen3-8B_train_seed42_original \
#    HF_REPO_ID=your-name/your-repo \
#    HF_PRIVATE=1 \
#    HF_COMMIT_MESSAGE="upload lora adapter" \
#    bash scripts/upload_RQ3_sft_checkpoint_to_hf.sh
#
#  Notes:
#    - Default UPLOAD_MODE=adapter_only: upload LoRA adapter + tokenizer files
#    - UPLOAD_MODE=all uploads the whole final directory
#    - Authentication prefers HF_TOKEN, then HUGGINGFACE_HUB_TOKEN
#    - README.md is generated automatically as a model card
# =============================================================

set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-checkpoints/RQ3_SFT_Qwen3-8B_train_seed42_original}"
HF_REPO_ID="${HF_REPO_ID:-PDP-Qwen3-8B-SFT-Lora}"
HF_PRIVATE="${HF_PRIVATE:-0}"
HF_REVISION="${HF_REVISION:-main}"
PATH_IN_REPO="${PATH_IN_REPO:-}"
REPO_TYPE="${REPO_TYPE:-model}"
UPLOAD_MODE="${UPLOAD_MODE:-adapter_only}"   # adapter_only | all
HF_COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-Upload RQ3 SFT LoRA adapter}"
GENERATE_MODEL_CARD="${GENERATE_MODEL_CARD:-1}"
HF_BASE_MODEL="${HF_BASE_MODEL:-Qwen/Qwen3-8B}"
HF_LICENSE="${HF_LICENSE:-other}"
HF_LANGUAGE="${HF_LANGUAGE:-zh}"
HF_MODEL_CARD_TITLE="${HF_MODEL_CARD_TITLE:-PDP Qwen3-8B RQ3 SFT LoRA Adapter}"
HF_DATASET_NAME="${HF_DATASET_NAME:-PDP-Bench / pdp2k_rq3_sft}"
HF_TAGS="${HF_TAGS:-legal llm chinese qwen3 peft lora prosecution-decision-prediction sft}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "pytorch" ]]; then
    echo "警告: 当前 conda 环境不是 pytorch (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset})。"
    echo "      huggingface_hub 通常在本地 pytorch 环境内。"
fi

if [[ -z "$HF_REPO_ID" ]]; then
    echo "错误: 必须设置 HF_REPO_ID，例如 your-name/PDP-Qwen3-8B-RQ3-SFT-LoRA"
    exit 1
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "错误: 待上传目录不存在: $LOCAL_DIR"
    exit 1
fi

if [[ "$UPLOAD_MODE" != "all" && "$UPLOAD_MODE" != "adapter_only" ]]; then
    echo "错误: UPLOAD_MODE 只能是 all 或 adapter_only，当前值: $UPLOAD_MODE"
    exit 1
fi

TOKEN_SOURCE="none"
if [[ -n "${HF_TOKEN:-}" ]]; then
    TOKEN_SOURCE="HF_TOKEN"
elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    TOKEN_SOURCE="HUGGINGFACE_HUB_TOKEN"
fi

echo "============================================================="
echo "Upload RQ3 SFT LoRA adapter to Hugging Face"
echo "  local_dir:       $LOCAL_DIR"
echo "  repo_id:         $HF_REPO_ID"
echo "  repo_type:       $REPO_TYPE"
echo "  private:         $HF_PRIVATE"
echo "  revision:        $HF_REVISION"
echo "  path_in_repo:    ${PATH_IN_REPO:-<repo-root>}"
echo "  upload_mode:     $UPLOAD_MODE"
echo "  token_source:    $TOKEN_SOURCE"
echo "============================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] command not executed."
    exit 0
fi

export LOCAL_DIR HF_REPO_ID HF_PRIVATE HF_REVISION PATH_IN_REPO REPO_TYPE UPLOAD_MODE HF_COMMIT_MESSAGE
export GENERATE_MODEL_CARD HF_BASE_MODEL HF_LICENSE HF_LANGUAGE HF_MODEL_CARD_TITLE HF_DATASET_NAME HF_TAGS

python - <<'PY'
import fnmatch
import os
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing huggingface_hub or tqdm. Activate the pytorch environment first."
    ) from exc


def as_bool(raw: str) -> bool:
    return raw == "1"


def should_ignore(relative_path: str, patterns: list[str]) -> bool:
    basename = relative_path.rsplit("/", 1)[-1]
    for pattern in patterns:
        if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(basename, pattern):
            return True
    return False


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


local_dir = os.environ["LOCAL_DIR"]
repo_id = os.environ["HF_REPO_ID"]
private = as_bool(os.environ["HF_PRIVATE"])
revision = os.environ["HF_REVISION"]
path_in_repo = os.environ.get("PATH_IN_REPO", "")
repo_type = os.environ["REPO_TYPE"]
upload_mode = os.environ["UPLOAD_MODE"]
commit_message = os.environ["HF_COMMIT_MESSAGE"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
generate_model_card = as_bool(os.environ["GENERATE_MODEL_CARD"])
base_model = os.environ["HF_BASE_MODEL"]
license_name = os.environ["HF_LICENSE"]
language = os.environ["HF_LANGUAGE"]
model_card_title = os.environ["HF_MODEL_CARD_TITLE"]
dataset_name = os.environ["HF_DATASET_NAME"]
tags = [tag for tag in os.environ["HF_TAGS"].split() if tag]

ADAPTER_ONLY_BASENAMES = {
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}

TRAINING_ARTIFACT_PATTERNS = [
    "checkpoint-*",
    "checkpoint-*/*",
    "global_step*",
    "global_step*/*",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
    "rng_state_*.pth",
    "events.out.tfevents.*",
]


def should_include_file(relative_path: str, upload_mode: str) -> bool:
    basename = relative_path.rsplit("/", 1)[-1]
    if relative_path == "README.md":
        return False
    if upload_mode == "all":
        return True
    if should_ignore(relative_path, TRAINING_ARTIFACT_PATTERNS):
        return False
    return basename in ADAPTER_ONLY_BASENAMES


def build_model_card(*, repo_id: str, local_dir_name: str, upload_mode: str) -> str:
    tags_yaml = "\n".join(f"- {tag}" for tag in tags) if tags else "- legal"
    checkpoint_note = ""
    if local_dir_name.startswith("checkpoint-"):
        checkpoint_note = (
            "\nThis repository was uploaded from an intermediate training checkpoint directory. "
            "For LoRA-based training, prefer uploading the final adapter output directory instead "
            "of a DeepSpeed inner checkpoint.\n"
        )
    return f"""---
license: {license_name}
language:
- {language}
base_model:
- {base_model}
tags:
{tags_yaml}
library_name: peft
pipeline_tag: text-generation
---

# {model_card_title}

## Model Summary

This repository contains an RQ3 supervised fine-tuning LoRA adapter for prosecution decision prediction (PDP) experiments based on `{base_model}`.

- Repository: `{repo_id}`
- Source directory: `{local_dir_name}`
- Upload mode: `{upload_mode}`
- Dataset: `{dataset_name}`

## Task

Given suspect information, procedural information, and factual information, the model is trained to generate structured prosecutorial reasoning and a final decision in the PDP setting.

The target decision space contains four labels:

- `起诉`
- `相对不起诉`
- `法定不起诉`
- `存疑不起诉`

## Training Output Format

The supervised target follows the native Qwen3-style format used in this project:

```text
<think>
...
</think>

【适用法条】
刑法：...
刑事诉讼法：...
刑事诉讼规则：...

【审查分析】
...

【最终结论】
决定：...
```

## Loading

This repository stores a LoRA adapter, not a merged standalone base model. Load it together with `{base_model}`, or merge it into the base model before standalone inference or vLLM evaluation.

## Notes
{checkpoint_note if checkpoint_note else ""}
This model card was generated automatically by `scripts/upload_RQ3_sft_checkpoint_to_hf.sh`.
"""


local_dir_path = Path(local_dir).resolve()
if not local_dir_path.is_dir():
    raise SystemExit(f"Local directory does not exist: {local_dir_path}")

files: list[tuple[Path, str, int]] = []
for file_path in sorted(p for p in local_dir_path.rglob("*") if p.is_file()):
    relative_path = file_path.relative_to(local_dir_path).as_posix()
    if not should_include_file(relative_path, upload_mode):
        continue
    files.append((file_path, relative_path, file_path.stat().st_size))

temp_model_card_path: Path | None = None
if generate_model_card:
    model_card_text = build_model_card(
        repo_id=repo_id,
        local_dir_name=local_dir_path.name,
        upload_mode=upload_mode,
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as tmp:
        tmp.write(model_card_text)
        temp_model_card_path = Path(tmp.name)
    files.append((temp_model_card_path, "README.md", temp_model_card_path.stat().st_size))

if not files:
    raise SystemExit("No files selected for upload. Check LOCAL_DIR or UPLOAD_MODE.")

total_bytes = sum(size for _, _, size in files)
base_repo_prefix = path_in_repo.strip("/")

api = HfApi(token=token)
print("[1/3] Creating or checking Hugging Face repo...")
try:
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
except HfHubHTTPError as exc:
    raise SystemExit(f"create_repo failed: {exc}") from exc

print("[2/3] Scanning files...")
print(f"  files: {len(files)}")
print(f"  total_size: {format_size(total_bytes)}")

print("[3/3] Uploading...")
progress = tqdm(files, total=len(files), unit="file", desc="Uploading", dynamic_ncols=True)
uploaded_bytes = 0
last_commit_info = None
for file_path, relative_path, size in progress:
    repo_relative_path = relative_path if not base_repo_prefix else f"{base_repo_prefix}/{relative_path}"
    progress.set_postfix_str(f"{format_size(uploaded_bytes)}/{format_size(total_bytes)}")
    print(f"Uploading: {relative_path} ({format_size(size)}) -> {repo_relative_path}")
    try:
        last_commit_info = api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=repo_relative_path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
        )
    except HfHubHTTPError as exc:
        raise SystemExit(f"upload_file failed: {relative_path}: {exc}") from exc
    uploaded_bytes += size
    progress.set_postfix_str(f"{format_size(uploaded_bytes)}/{format_size(total_bytes)}")

progress.close()
print("Upload complete.")
if last_commit_info is not None:
    print(last_commit_info)
PY
