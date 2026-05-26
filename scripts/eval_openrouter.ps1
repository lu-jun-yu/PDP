# =============================================================
#  scripts/eval_openrouter.ps1 — PDP 评估启动脚本 (OpenRouter API)
#
#  用法:
#    .\scripts\eval_openrouter.ps1
#    .\scripts\eval_openrouter.ps1 -Variants original,definitions,one_shot
#
#  前置条件:
#    1. 设置环境变量 OPENROUTER_API_KEY
#    2. 安装依赖: pip install openai datasets
# =============================================================

param(
    [string[]]$Variants = @("original", "definitions", "one_shot"),
    [ValidateSet("auto", "none", "low", "medium", "high")]
    [string]$ReasoningEffort = "auto",
    [switch]$NoResume
)

# ---- API Key ----
if (-not $env:OPENROUTER_API_KEY) {
    $envFile = Join-Path $PSScriptRoot "..\.env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match '^\s*OPENROUTER_API_KEY\s*=\s*(.+)$') {
                $env:OPENROUTER_API_KEY = $Matches[1].Trim()
            }
        }
    }
}

# 检查 API Key
if (-not $env:OPENROUTER_API_KEY) {
    Write-Host "错误: 请设置 OPENROUTER_API_KEY 环境变量" -ForegroundColor Red
    Write-Host '  方式一: $env:OPENROUTER_API_KEY = "sk-or-v1-..."'
    Write-Host "  方式二: 在项目根目录创建 .env 文件，写入 OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
}

# ---- 参数配置 ----
$MODEL       = "deepseek/deepseek-v4-pro"
$DATA_PATH   = "data/pdp4k"
$MAX_TOKENS  = 4096
$TEMPERATURE = 1.0
$TOP_P       = 0.95
$TOP_K       = 20
$MIN_P       = 0.0
$CONCURRENCY = 100
$BATCH_SIZE  = 100
$OUTPUT_DIR  = "results"

$VALID_VARIANTS = @("original", "definitions", "one_shot")
foreach ($variant in $Variants) {
    if ($variant -notin $VALID_VARIANTS) {
        Write-Host "错误: 未知 prompt variant: $variant" -ForegroundColor Red
        Write-Host "可选值: $($VALID_VARIANTS -join ', ')"
        exit 1
    }
}

# ---- 运行三种消融实验 ----
foreach ($variant in $Variants) {
    Write-Host "=============================================================" -ForegroundColor Cyan
    Write-Host "运行消融实验: $variant" -ForegroundColor Cyan
    Write-Host "=============================================================" -ForegroundColor Cyan

    $cmdArgs = @(
        "eval/evaluate_openrouter.py",
        "--model", $MODEL,
        "--data-path", $DATA_PATH,
        "--max-tokens", $MAX_TOKENS,
        "--concurrency", $CONCURRENCY,
        "--batch-size", $BATCH_SIZE,
        "--output-dir", $OUTPUT_DIR,
        "--prompt-variant", $variant,
        "--reasoning-effort", $ReasoningEffort
        # "--temperature", $TEMPERATURE,
        # "--top-p", $TOP_P,
        # "--top-k", $TOP_K,
        # "--min-p", $MIN_P
    )

    if ($NoResume) {
        $cmdArgs += "--no-resume"
    }

    python @cmdArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
