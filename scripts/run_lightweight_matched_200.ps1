param([int]$InitialWorkerPid = 0)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$compat = Join-Path $root '.venv_minicpm4_compat\Scripts\python.exe'
$out = Join-Path $root 'outputs\evaluation\local_models_200_256'
if ($InitialWorkerPid -gt 0) {
    $initial = Get-Process -Id $InitialWorkerPid -ErrorAction SilentlyContinue
    if ($initial) { $initial.WaitForExit() }
}
& $python scripts/sync_lightweight_comparison.py *> (Join-Path $out 'manuscript_sync_initial.log')
& $python -u scripts/run_lightweight_without_minicpm.py *> (Join-Path $out 'campaign.log')
$evaluationCode = $LASTEXITCODE
& $python scripts/sync_lightweight_comparison.py *> (Join-Path $out 'manuscript_sync_final.log')
if ($LASTEXITCODE -ne 0) { throw 'Manuscript synchronization failed; see manuscript_sync_final.log' }
Set-Location -LiteralPath (Join-Path $root 'manuscript')
& 'G:\tools\texlive\2026\bin\windows\latexmk.exe' -pdf -interaction=nonstopmode -halt-on-error sn-article.tex *> evidence/lightweight_matched_build.log
if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed; see evidence/lightweight_matched_build.log' }
Write-Output "Evaluation campaign exited $evaluationCode; manuscript synchronized and compiled."
