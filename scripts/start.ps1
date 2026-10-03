$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
& '.venv\Scripts\python.exe' -B -m experiment_core.launch start
exit $LASTEXITCODE
