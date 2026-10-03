param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Run scripts/setup.ps1 first.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
& $PythonPath -B -m experiment_core.cli @Arguments
exit $LASTEXITCODE
