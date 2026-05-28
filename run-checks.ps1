param(
    [switch]$RunEvaluation,
    [int]$EvaluationMaxCases = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$targetScript = Join-Path $repoRoot "scripts\run_local_checks.ps1"

if (-not (Test-Path $targetScript)) {
    throw "Target script not found: $targetScript"
}

& $targetScript -RunEvaluation:$RunEvaluation -EvaluationMaxCases $EvaluationMaxCases
