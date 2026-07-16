#!/usr/bin/env pwsh
# =============================================================
# scripts/eval_RQ2_1_resample.ps1 - RQ2.1 rebuttal resample eval (OpenRouter)
#
# Run all requested model / reasoning-budget combinations.
# Results are organized as:
#   results/RQ2_1_resample/<experiment>/<model_display>/<budget_label>/
#
# For each <budget_label> directory, the following are saved together:
#   - split-level result directories for resample_1 / resample_2 / resample_3
#   - one markdown summary
#   - one json summary
#
# Budget semantics:
#   - effort budgets: none / low / medium / high / xhigh
#   - numeric budgets: 0 / 512 / 1024 / ...
#   - numeric budget 0 means "reasoning disabled", implemented via
#     --reasoning-effort none because evaluate_openrouter_rq2.py does not
#     accept --reasoning-max-tokens 0.
#
# Usage:
#   .\scripts\eval_RQ2_1_resample.ps1
#   .\scripts\eval_RQ2_1_resample.ps1 -Experiment charge_coverage
# =============================================================

param(
    [ValidateSet("major_group_decision_balance", "charge_coverage")]
    [string]$Experiment = "major_group_decision_balance",
    [string]$ReuseRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$VARIANT = "original"
$SPLITS = @("resample_1", "resample_2", "resample_3")
$MAX_TOKENS = 16384
$CONCURRENCY = 20
$NUM_REPEATS = 1

$MODEL_CONFIGS = @(
    [PSCustomObject]@{
        DisplayName = "GPT-5.4"
        ModelId     = "openai/gpt-5.4"
        BudgetMode  = "effort"
        Budgets     = @("none", "low", "medium", "high", "xhigh")
    },
    [PSCustomObject]@{
        DisplayName = "Gemini-3.1-Pro"
        ModelId     = "google/gemini-3.1-pro-preview"
        BudgetMode  = "effort"
        Budgets     = @("low", "medium", "high")
    },
    [PSCustomObject]@{
        DisplayName = "Claude-Opus-4.6"
        ModelId     = "anthropic/claude-opus-4.6"
        BudgetMode  = "max_tokens"
        Budgets     = @(0, 1024, 2048, 4096)
    },
    [PSCustomObject]@{
        DisplayName = "DeepSeek-V4-Pro"
        ModelId     = "deepseek/deepseek-v4-pro"
        BudgetMode  = "effort"
        Budgets     = @("none", "high", "xhigh")
    },
    [PSCustomObject]@{
        DisplayName = "Qwen3.6-Max"
        ModelId     = "qwen/qwen3.6-max-preview"
        BudgetMode  = "max_tokens"
        Budgets     = @(0, 512, 1024, 2048, 4096)
    }
)

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

$DATA_PATH = if ($Experiment -eq "charge_coverage") {
    "data_process/Rebuttal/Resample/pdp_bench_rq2_charge_coverage"
} else {
    "data_process/Rebuttal/Resample/pdp_bench_rq2_major_group_decision_balance"
}
$RESULT_ROOT = Join-Path "results/RQ2_1_resample" $Experiment

function New-BudgetArgumentList {
    param(
        [string]$ModelId,
        [string]$DataPath,
        [string[]]$Splits,
        [string]$OutputDir,
        [string]$PromptVariant,
        [int]$MaxTokens,
        [int]$Concurrency,
        [int]$NumRepeats,
        [string]$BudgetMode,
        [object]$Budget
    )

    $argList = @(
        "data_process/Rebuttal/Resample/run_rq2_resample_budget.py",
        "--model", $ModelId,
        "--data-path", $DataPath,
        "--splits"
    ) + $Splits + @(
        "--num-repeats", "$NumRepeats",
        "--max-tokens", "$MaxTokens",
        "--concurrency", "$Concurrency",
        "--output-dir", $OutputDir,
        "--prompt-variant", $PromptVariant
    )

    if ($BudgetMode -eq "effort") {
        $argList += @("--reasoning-effort", [string]$Budget)
    } elseif ($BudgetMode -eq "max_tokens") {
        $budgetInt = [int]$Budget
        if ($budgetInt -eq 0) {
            $argList += @("--reasoning-effort", "none")
        } else {
            $argList += @("--reasoning-max-tokens", "$budgetInt")
        }
    } else {
        throw "Unsupported BudgetMode: $BudgetMode"
    }

    return $argList
}

function Test-BudgetCompleted {
    param(
        [string]$BudgetDir
    )

    if (-not (Test-Path -LiteralPath $BudgetDir)) {
        return $false
    }

    $md = @(Get-ChildItem -LiteralPath $BudgetDir -Filter "*_summary.md" -File -ErrorAction SilentlyContinue)
    $json = @(Get-ChildItem -LiteralPath $BudgetDir -Filter "*_summary.json" -File -ErrorAction SilentlyContinue)
    return ($md.Count -gt 0 -and $json.Count -gt 0)
}

function Test-BudgetHasCheckpoint {
    param(
        [string]$BudgetDir
    )

    if (-not (Test-Path -LiteralPath $BudgetDir)) {
        return $false
    }

    $ckpts = @(Get-ChildItem -LiteralPath $BudgetDir -Filter "*_checkpoint.jsonl" -File -ErrorAction SilentlyContinue)
    return ($ckpts.Count -gt 0)
}

Write-Host "============================================================="
Write-Host "RQ2.1 rebuttal resample evaluation"
Write-Host "experiment: $Experiment"
Write-Host "dataset: $DATA_PATH"
Write-Host "prompt variant: $VARIANT"
Write-Host "splits: $($SPLITS -join ' '), repeats per sample: $NUM_REPEATS"
Write-Host "models:"
foreach ($cfg in $MODEL_CONFIGS) {
    Write-Host "  - $($cfg.DisplayName): $($cfg.Budgets -join ' ')"
}
Write-Host "============================================================="

$overallFail = $false

foreach ($cfg in $MODEL_CONFIGS) {
    Write-Host ""
    Write-Host "-------------------------------------------------------------"
    Write-Host "model: $($cfg.DisplayName) ($($cfg.ModelId))"
    Write-Host "budgets: $($cfg.Budgets -join ' ')"
    Write-Host "-------------------------------------------------------------"

    $modelRoot = Join-Path $RESULT_ROOT $cfg.DisplayName
    $procs = @()

    foreach ($budget in $cfg.Budgets) {
        $budgetLabel = [string]$budget
        $budgetOutputDir = Join-Path $modelRoot $budgetLabel
        $reuseBudgetDir = $null
        if ($ReuseRoot) {
            $candidateReuseDir = Join-Path (Join-Path $ReuseRoot $cfg.DisplayName) $budgetLabel
            if (Test-Path -LiteralPath $candidateReuseDir) {
                $reuseBudgetDir = $candidateReuseDir
            }
        }

        if (Test-BudgetCompleted -BudgetDir $budgetOutputDir) {
            Write-Host "[skip] model=$($cfg.DisplayName) budget=$budgetLabel (summary exists)"
            continue
        }

        $hasCheckpoint = Test-BudgetHasCheckpoint -BudgetDir $budgetOutputDir
        $argList = New-BudgetArgumentList `
            -ModelId $cfg.ModelId `
            -DataPath $DATA_PATH `
            -Splits $SPLITS `
            -OutputDir $budgetOutputDir `
            -PromptVariant $VARIANT `
            -MaxTokens $MAX_TOKENS `
            -Concurrency $CONCURRENCY `
            -NumRepeats $NUM_REPEATS `
            -BudgetMode $cfg.BudgetMode `
            -Budget $budget
        if ($reuseBudgetDir) {
            $argList += @("--reuse-dir", $reuseBudgetDir)
        }

        if ($cfg.BudgetMode -eq "effort") {
            if ($hasCheckpoint) {
                Write-Host "[resume] model=$($cfg.DisplayName) effort=$budgetLabel"
            } elseif ($reuseBudgetDir) {
                Write-Host "[start] model=$($cfg.DisplayName) effort=$budgetLabel reuse=$reuseBudgetDir"
            } else {
                Write-Host "[start] model=$($cfg.DisplayName) effort=$budgetLabel"
            }
        } elseif ([int]$budget -eq 0) {
            if ($hasCheckpoint) {
                Write-Host "[resume] model=$($cfg.DisplayName) budget=0 (reasoning disabled)"
            } elseif ($reuseBudgetDir) {
                Write-Host "[start] model=$($cfg.DisplayName) budget=0 (reasoning disabled) reuse=$reuseBudgetDir"
            } else {
                Write-Host "[start] model=$($cfg.DisplayName) budget=0 (reasoning disabled)"
            }
        } else {
            if ($hasCheckpoint) {
                Write-Host "[resume] model=$($cfg.DisplayName) max_tokens=$budgetLabel"
            } elseif ($reuseBudgetDir) {
                Write-Host "[start] model=$($cfg.DisplayName) max_tokens=$budgetLabel reuse=$reuseBudgetDir"
            } else {
                Write-Host "[start] model=$($cfg.DisplayName) max_tokens=$budgetLabel"
            }
        }

        $proc = Start-Process -FilePath "python" -ArgumentList $argList -NoNewWindow -PassThru
        $procs += [PSCustomObject]@{
            Name      = "$($cfg.DisplayName)/$budgetLabel"
            Process   = $proc
            BudgetDir = $budgetOutputDir
        }
    }

    foreach ($item in $procs) {
        $null = $item.Process.WaitForExit()
        $item.Process.Refresh()
        $code = $item.Process.ExitCode
        $completed = Test-BudgetCompleted -BudgetDir $item.BudgetDir
        if ($completed) {
            Write-Host "[done] $($item.Name)"
        } elseif ($null -ne $code -and "$code" -ne "" -and $code -eq 0) {
            Write-Host "[done] $($item.Name)"
        } else {
            Write-Host "[failed] $($item.Name) exit_code=$code" -ForegroundColor Red
            $overallFail = $true
        }
    }
}

$summaryArgs = @(
    "data_process/Rebuttal/Resample/build_experiment_summary.py",
    "--experiment-root", $RESULT_ROOT,
    "--output-md", (Join-Path $RESULT_ROOT "summary.md"),
    "--title", "RQ2.1 Rebuttal Resample - $Experiment"
)
& python @summaryArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build experiment summary markdown." -ForegroundColor Red
    $overallFail = $true
}

if ($overallFail) {
    Write-Host "Some rebuttal resample evaluations failed. Please check logs." -ForegroundColor Red
    exit 1
}

Write-Host "All rebuttal resample evaluations completed."
