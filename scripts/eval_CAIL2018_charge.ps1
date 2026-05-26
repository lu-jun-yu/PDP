#!/usr/bin/env pwsh
# =============================================================
# scripts/eval_CAIL2018_charge.ps1 - CAIL2018 charge eval for RQ1
#
# Usage:
#   .\scripts\eval_CAIL2018_charge.ps1
# =============================================================

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$REASONING_EFFORT = "auto"

if ($args.Count -ne 0) {
    Write-Host "Error: scripts/eval_CAIL2018_charge.ps1 does not accept arguments." -ForegroundColor Red
    Write-Host "Usage: .\scripts\eval_CAIL2018_charge.ps1"
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

# ---- Config aligned with scripts/eval_RQ1.ps1 ----
$MODELS = @(
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-opus-4.6",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.6-max-preview",
    "openai/gpt-oss-20b",
    "qwen/qwen3.5-35b-a3b"
)
$INPUT_FILE = "data/CAIL2018/exercise_contest/data_test_charge_4k_seed42.json"
$MAX_TOKENS = 8192
$CONCURRENCY = 20
$BATCH_SIZE = 100
$OUTPUT_DIR = "results/RQ1_CAIL2018_charge"

Write-Host "============================================================="
Write-Host "CAIL2018-Small charge prediction for RQ1"
Write-Host "models: $($MODELS -join ' ')"
Write-Host "input file: $INPUT_FILE"
Write-Host "reasoning-effort: $REASONING_EFFORT"
Write-Host "============================================================="
Write-Host ""
Write-Host "-------------------------------------------------------------"
Write-Host "start all models in parallel"
Write-Host "-------------------------------------------------------------"

$procs = @()
foreach ($MODEL in $MODELS) {
    Write-Host "[start] model=$MODEL"
    $argList = @(
        "eval/evaluate_cail_charge_openrouter.py",
        "--model", $MODEL,
        "--input-file", $INPUT_FILE,
        "--max-tokens", "$MAX_TOKENS",
        "--concurrency", "$CONCURRENCY",
        "--batch-size", "$BATCH_SIZE",
        "--output-dir", $OUTPUT_DIR,
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
    Write-Host "Some model evaluations failed. Please check logs." -ForegroundColor Red
    exit 1
}

Write-Host "All CAIL2018 charge evaluations completed."
