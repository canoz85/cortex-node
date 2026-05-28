param(
    [switch]$RunEvaluation,
    [int]$EvaluationMaxCases = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Create/activate your venv first."
}

Push-Location $repoRoot
try {
    Write-Host "[1/4] Running tests with coverage JSON..." -ForegroundColor Cyan
    & $pythonExe -m pytest --cov-report=json:coverage.json

    Write-Host "[2/4] Running coverage ratchet check..." -ForegroundColor Cyan
    & $pythonExe scripts/coverage_ratchet.py `
        --coverage-json coverage.json `
        --policy .github/coverage-policy.json `
        --summary-out benchmarks/results/coverage-ratchet.md

    Write-Host "[3/4] Running benchmark skeleton smoke..." -ForegroundColor Cyan
    & $pythonExe scripts/benchmark.py `
        --cases benchmarks/scenarios.json `
        --max-cases 1 `
        --output benchmarks/results/local-skeleton.json

    Write-Host "[4/4] Building benchmark trend report..." -ForegroundColor Cyan
    & $pythonExe scripts/benchmark_trend.py `
        --results-dir benchmarks/results `
        --output benchmarks/results/trend.md

    if ($RunEvaluation) {
        Write-Host "[extra] Running evaluation dataset and scoring dashboard..." -ForegroundColor Cyan
        & $pythonExe scripts/run_evaluation.py `
            --live `
            --max-cases $EvaluationMaxCases `
            --output benchmarks/results/evaluation-latest.json `
            --dashboard-md benchmarks/results/evaluation-dashboard.md `
            --dashboard-json benchmarks/results/evaluation-dashboard.json
    }

    Write-Host "Local checks passed." -ForegroundColor Green
}
finally {
    Pop-Location
}
