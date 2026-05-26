#!/usr/bin/env pwsh
# =============================================================
# scripts/eval_RQ2_1.ps1 - RQ2.1 parallel reasoning-budget eval (OpenRouter)
#
# Single model; start one process per reasoning budget setting (test_rq2 split, 100 cases),
# with 5 repeated runs per case.
#
# Budget mode (mutually exclusive at API level):
#   - "effort": pass --reasoning-effort for each entry in REASONING_EFFORTS
#   - "max_tokens": pass --reasoning-max-tokens for each entry in REASONING_MAX_TOKENS_LIST
#
# Usage:
#   .\scripts\eval_RQ2_1.ps1
# =============================================================

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$MODEL = "qwen/qwen3.6-max-preview"
$VARIANT = "original"

# "effort" | "max_tokens"
$REASONING_BUDGET_MODE = "max_tokens"
$REASONING_EFFORTS = @("none") # "none", "low", "medium", "high", "xhigh"
# OpenRouter 要求 completion 的 --max-tokens 严格大于 reasoning.max_tokens；勿让下列最大值 >= $MAX_TOKENS。
$REASONING_MAX_TOKENS_LIST = @(2048, 4096)

if ($args.Count -ne 0) {
    Write-Host "Error: scripts/eval_RQ2_1.ps1 does not accept arguments." -ForegroundColor Red
    Write-Host "Usage: .\scripts\eval_RQ2_1.ps1"
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
$OUTPUT_DIR = "results/RQ2_1"
$NUM_REPEATS = 8

Write-Host "============================================================="
Write-Host "RQ2.1 parallel evaluation (single model, different reasoning budgets)"
Write-Host "model: $MODEL"
Write-Host "prompt variant: $VARIANT"
Write-Host "reasoning budget mode: $REASONING_BUDGET_MODE"
if ($REASONING_BUDGET_MODE -eq "effort") {
    Write-Host "reasoning efforts: $($REASONING_EFFORTS -join ' ')"
} elseif ($REASONING_BUDGET_MODE -eq "max_tokens") {
    Write-Host "reasoning max_tokens: $($REASONING_MAX_TOKENS_LIST -join ' ')"
} else {
    Write-Host "Error: REASONING_BUDGET_MODE must be 'effort' or 'max_tokens'." -ForegroundColor Red
    exit 1
}
Write-Host "split: $SPLIT, repeats per sample: $NUM_REPEATS"
Write-Host "============================================================="
Write-Host ""
Write-Host "-------------------------------------------------------------"
Write-Host "start all reasoning budget runs in parallel"
Write-Host "-------------------------------------------------------------"

$procs = @()

if ($REASONING_BUDGET_MODE -eq "effort") {
    foreach ($EFFORT in $REASONING_EFFORTS) {
        Write-Host "[start] model=$MODEL variant=$VARIANT reasoning-effort=$EFFORT"
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
            "--reasoning-effort", $EFFORT
        )
        $proc = Start-Process -FilePath "python" -ArgumentList $argList -NoNewWindow -PassThru
        $procs += [PSCustomObject]@{
            Name    = "effort=$EFFORT"
            Process = $proc
        }
    }
} else {
    foreach ($MT in $REASONING_MAX_TOKENS_LIST) {
        Write-Host "[start] model=$MODEL variant=$VARIANT reasoning-max-tokens=$MT"
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
            "--reasoning-max-tokens", "$MT"
        )
        $proc = Start-Process -FilePath "python" -ArgumentList $argList -NoNewWindow -PassThru
        $procs += [PSCustomObject]@{
            Name    = "max_tokens=$MT"
            Process = $proc
        }
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

Write-Host "All reasoning budget evaluations completed."
