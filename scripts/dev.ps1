$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
Push-Location frontend
try { npm run build -- --watch } finally { Pop-Location }
