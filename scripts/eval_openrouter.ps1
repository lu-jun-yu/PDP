# =============================================================
#  scripts/eval_openrouter.ps1 — PDP 评估启动脚本 (OpenRouter API)
#
#  用法:
#    .\scripts\eval_openrouter.ps1
#
#  前置条件:
#    1. 设置环境变量 OPENROUTER_API_KEY
#    2. 安装依赖: pip install openai datasets
# =============================================================

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

# ---- 参数配置（直接在此修改） ----
$MODEL       = "deepseek/deepseek-v3.2"
$DATA_PATH   = "data/pdp25k"
$MAX_TOKENS  = 2048
$TEMPERATURE = 1.0
$TOP_P       = 0.95
$TOP_K       = 20
$MIN_P       = 0.0
$CONCURRENCY = 5                   # 并发请求数
$BATCH_SIZE  = 5
$OUTPUT_DIR  = "results"

# # ---- 运行评估 (baseline) ----
# python eval/evaluate_openrouter.py `
#     --model $MODEL `
#     --data-path $DATA_PATH `
#     --max-tokens $MAX_TOKENS `
#     --temperature $TEMPERATURE `
#     --top-p $TOP_P `
#     --top-k $TOP_K `
#     --min-p $MIN_P `
#     --concurrency $CONCURRENCY `
#     --batch-size $BATCH_SIZE `
#     --output-dir $OUTPUT_DIR

# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 运行评估 (with definitions) ----
python eval/evaluate_openrouter.py `
    --model $MODEL `
    --data-path $DATA_PATH `
    --max-tokens $MAX_TOKENS `
    --concurrency $CONCURRENCY `
    --batch-size $BATCH_SIZE `
    --output-dir $OUTPUT_DIR `
    --with-definitions `
    # --temperature $TEMPERATURE `
    # --top-p $TOP_P `
    # --top-k $TOP_K `
    # --min-p $MIN_P `
