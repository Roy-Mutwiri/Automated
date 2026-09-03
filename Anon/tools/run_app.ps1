<#
.SYNOPSIS
    Launch the presenter from THIS worktree, whatever the venv thinks.

.DESCRIPTION
    The shared .venv contains an editable install of `presenter` pointing at
    whichever worktree it was installed from. Running `-m presenter.app`
    directly can therefore execute another branch's source while your edits sit
    on disk doing nothing - a silent and genuinely confusing failure.

    PYTHONPATH is set here so this worktree's src/ wins, and --debug prints the
    source root it actually loaded so the answer is always on screen.

.EXAMPLE
    tools\run_app.ps1                        # camera angles, no GPU
    tools\run_app.ps1 -Renderer liveportrait # photoreal
#>
param(
    [ValidateSet('camera-preview', 'schematic', 'liveportrait')]
    [string] $Renderer = 'camera-preview',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Rest
)

$Here = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Here 'src'
$python = Join-Path $Here '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Error "no interpreter at $python"
    exit 1
}

& $python -m presenter.app --renderer $Renderer --debug @Rest
