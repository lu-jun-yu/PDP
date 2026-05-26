#!/usr/bin/env pwsh
# =============================================================
# scripts/eval_RQ2_3.ps1 - RQ2.3 parallel prompt-variant eval (OpenRouter)
#
# 固定模型；对 test_rq2 split 上每一种 prompt variant 各启一个进程；
# reasoning 使用默认配置 auto（不显式控制 effort / max_tokens，由路由与模型默认决定）。
# 每条样本重复 NUM_REPEATS 次（默认 8）。
#
# 变体列表与 eval / prompt_template.PROMPT_VARIANTS 一致：
#   original, definitions, one_shot
#
# Usage:
#   .\scripts\eval_RQ2_3.ps1
# =============================================================

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$MODEL = "qwen/qwen3.6-max-preview"

# 与 prompt_template.PROMPT_VARIANTS 保持同步
$VARIANTS = @("original", "definitions", "one_shot")

if ($args.Count -ne 0) {
    Write-Host "Error: scripts/eval_RQ2_3.ps1 does not accept arguments." -ForegroundColor Red
    Write-Host "Usage: .\scripts\eval_RQ2_3.ps1"
    exit 1
}

if (-not $env:OPENROUTER_API_KEY) {
    $envFile = Join-Path $PSScriptRoot "..\.env"
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            if ($line -match '^\s*OPENROUTER_API_KEY\s*=\s*(.+)$') {
                $env:OPENROUTER_API_KEY = $Matches[1].Trim()
                break
            }
        }
    }
}

if (-not $env:OPENROUTER_API_KEY) {
    Write-Host "Error: OPENROUTER_API_KEY is required." -ForegroundColor Red
    Write-Host '  Option 1: $env:OPENROUTER_API_KEY = "sk-or-v1-..."'
    Write-Host "  Option 2: create .env in repo root with OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
}

$DATA_PATH = "data/pdp4k"
$SPLIT = "test_rq2"
$MAX_TOKENS = 16384
$CONCURRENCY = 20
$BATCH_SIZE = 100
$OUTPUT_DIR = "results/RQ2_3"
$NUM_REPEATS = 8

Write-Host "============================================================="
Write-Host "RQ2.3 parallel evaluation (single model, all prompt variants)"
Write-Host "model: $MODEL"
Write-Host "prompt variants: $($VARIANTS -join ', ')"
Write-Host "reasoning-effort: auto (default routing)"
Write-Host "split: $SPLIT, repeats per sample: $NUM_REPEATS"
Write-Host "============================================================="
Write-Host ""
Write-Host "-------------------------------------------------------------"
Write-Host "start all prompt-variant runs in parallel"
Write-Host "-------------------------------------------------------------"

$procs = @()
foreach ($VARIANT in $VARIANTS) {
    Write-Host "[start] model=$MODEL variant=$VARIANT reasoning-effort=auto"
    $argList = @(
        "eval/evaluate_openrouter_rq2.py",
        "--model", $MODEL,
        "--data-path", $DATA_PATH,
        "--split", $SPLIT,
        "--num-repeats", "$NUM_REPEATS",
        "--max-tokens", "$MAX_TOKENS",
        "--concurrency", "$CONCURRENCY",
        "--batch-size", "$BATCH_SIZE",
        "--output-dir", $OUTPUT_DIR,
        "--prompt-variant", $VARIANT,
        "--reasoning-effort", "auto"
    )
    $proc = Start-Process -FilePath "python" -ArgumentList $argList -NoNewWindow -PassThru
    $procs += [PSCustomObject]@{
        Name    = "variant=$VARIANT"
        Process = $proc
    }
}

$overallFail = $false
foreach ($item in $procs) {
    $null = $item.Process.WaitForExit()
    $item.Process.Refresh()
    $code = $item.Process.ExitCode
    if ($code -eq 0) {
        Write-Host "[done] $($item.Name)"
    } else {
        Write-Host "[failed] $($item.Name) exit_code=$code" -ForegroundColor Red
        $overallFail = $true
    }
}

if ($overallFail) {
    Write-Host "Some evaluations failed. Please check logs." -ForegroundColor Red
    exit 1
}

Write-Host "All prompt-variant evaluations completed."
