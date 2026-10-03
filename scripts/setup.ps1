$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    python -m venv .bootstrap
    & '.bootstrap\Scripts\python.exe' -m pip install 'uv==0.9.5'
    if ($LASTEXITCODE -ne 0) { throw 'uv installation failed' }
    $UvPath = Join-Path $ProjectRoot '.bootstrap\Scripts\uv.exe'
} else { $UvPath = (Get-Command uv).Source }
& $UvPath sync --locked --group dev
if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed' }
Push-Location frontend
try { npm ci --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }; npm run build; if ($LASTEXITCODE -ne 0) { throw 'UI build failed' } } finally { Pop-Location }
Write-Output 'Ready. Run scripts/start.ps1.'
