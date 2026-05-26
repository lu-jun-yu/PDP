#!/usr/bin/env pwsh
# =============================================================
# scripts/eval_RQ1.ps1 - RQ1 parallel evaluation (OpenRouter API)
#
# Usage:
#   .\scripts\eval_RQ1.ps1
# =============================================================

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$VARIANT = "original"
$REASONING_EFFORT = "auto"

if ($args.Count -ne 0) {
    Write-Host "Error: scripts/eval_RQ1.ps1 does not accept arguments." -ForegroundColor Red
    Write-Host "Usage: .\scripts\eval_RQ1.ps1"
    exit 1
}

# ---- API Key ----
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

# ---- Config ----
$MODELS = @(
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-opus-4.6",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.6-max-preview",
    "openai/gpt-oss-20b",
    "qwen/qwen3.5-35b-a3b"
)
$DATA_PATH = "data/pdp4k"
$MAX_TOKENS = 8192
$CONCURRENCY = 20
$BATCH_SIZE = 100
$OUTPUT_DIR = "results/RQ1"

Write-Host "============================================================="
Write-Host "RQ1 parallel evaluation"
Write-Host "models: $($MODELS -join ' ')"
Write-Host "prompt variant: $VARIANT"
Write-Host "reasoning-effort: $REASONING_EFFORT"
Write-Host "============================================================="
Write-Host ""
Write-Host "-------------------------------------------------------------"
Write-Host "current variant: $VARIANT (start all models in parallel)"
Write-Host "-------------------------------------------------------------"

$procs = @()
foreach ($MODEL in $MODELS) {
    Write-Host "[start] model=$MODEL variant=$VARIANT"
    $argList = @(
        "eval/evaluate_openrouter.py",
        "--model", $MODEL,
        "--data-path", $DATA_PATH,
        "--max-tokens", "$MAX_TOKENS",
        "--concurrency", "$CONCURRENCY",
        "--batch-size", "$BATCH_SIZE",
        "--output-dir", $OUTPUT_DIR,
        "--prompt-variant", $VARIANT,
        "--reasoning-effort", $REASONING_EFFORT
    )
    $proc = Start-Process -FilePath "python" -ArgumentList $argList -NoNewWindow -PassThru
    $procs += [PSCustomObject]@{
        Name    = $MODEL
        Process = $proc
    }
}

$overallFail = $false
foreach ($item in $procs) {
    # WaitForExit 持有进程句柄，避免子进程已退出时 Wait-Process -Id 报「找不到进程」
    $null = $item.Process.WaitForExit()
    $item.Process.Refresh()
    $code = $item.Process.ExitCode
    if ($code -eq 0) {
        Write-Host "[done] $($item.Name) (variant=$VARIANT)"
    } else {
        Write-Host "[failed] $($item.Name) (variant=$VARIANT) exit_code=$code" -ForegroundColor Red
        $overallFail = $true
    }
}

if ($overallFail) {
    Write-Host "Some model evaluations failed. Please check logs." -ForegroundColor Red
    exit 1
}

Write-Host "All model evaluations completed."
