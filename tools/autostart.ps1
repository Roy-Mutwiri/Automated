<#
.SYNOPSIS
    Registers (or removes) one scheduled watcher per git worktree.

.DESCRIPTION
    Ownership is per branch, and a branch lives in a worktree, so the watcher is
    per worktree too. One task per worktree, each launching tools/sync.ps1 with
    -Worktree pointed at its own directory, so a watcher can only ever commit
    and push the branch it is sitting on.

    A single task watching the whole repo cannot do this: it would see all three
    worktrees as one tree, and be back to guessing an owner from file paths.

    Tasks are named AutomatedRepoSync_<worktree folder>. The legacy single task
    'AutomatedRepoSync' is removed on sight - it watched the shared repo with
    folder attribution, which is exactly the behaviour branch ownership
    replaced.

.PARAMETER Remove
    Unregister every AutomatedRepoSync task instead of creating them.

.PARAMETER Start
    Start each task immediately after registering, instead of waiting for the
    next logon.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\autostart.ps1 -Start
    powershell -ExecutionPolicy Bypass -File tools\autostart.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch] $Remove,
    [switch] $Start
)

$ErrorActionPreference = 'Stop'

$TaskPrefix = 'AutomatedRepoSync'
$LegacyTask = 'AutomatedRepoSync'
$SyncPath   = Join-Path $PSScriptRoot 'sync.ps1'
$RepoRoot   = Split-Path -Parent $PSScriptRoot

function Remove-SyncTask {
    param([string] $Name)
    try {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction Stop
        Write-Host "Removed scheduled task '$Name'."
    }
    catch { }
}

# Every worktree attached to this repo, including the shared one. Read from git
# rather than guessed from sibling folder names, so adding a worktree and
# re-running this script is all it takes to give it a watcher.
function Get-Worktrees {
    $paths = @()
    foreach ($line in (& git -C $RepoRoot worktree list --porcelain)) {
        if ($line -like 'worktree *') {
            $paths += ($line.Substring(9) -replace '/', '\')
        }
    }
    return $paths
}

if ($Remove) {
    Remove-SyncTask -Name $LegacyTask
    foreach ($task in @(Get-ScheduledTask -TaskName "$TaskPrefix*" -ErrorAction SilentlyContinue)) {
        Remove-SyncTask -Name $task.TaskName
    }
    return
}

if (-not (Test-Path $SyncPath)) { throw "sync.ps1 not found at $SyncPath" }

# The legacy task attributed by folder. Leaving it registered would run a second
# watcher over the shared repo alongside the per-worktree ones.
Remove-SyncTask -Name $LegacyTask

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Restart if it ever dies; never let Windows stop it for running "too long".
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited

foreach ($worktree in Get-Worktrees) {
    if (-not (Test-Path -LiteralPath $worktree)) {
        Write-Host "Skipping missing worktree $worktree."
        continue
    }

    $leaf     = Split-Path -Leaf $worktree
    $taskName = '{0}_{1}' -f $TaskPrefix, $leaf
    $branch   = (& git -C $worktree rev-parse --abbrev-ref HEAD | Select-Object -First 1)

    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
        '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -Worktree "{1}"' -f $SyncPath, $worktree
    )

    # -Force resets a task to enabled. A terminal that deliberately turned its
    # watcher off would have it silently turned back on by an unrelated re-run
    # of this script, so a disabled task stays disabled.
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    $wasDisabled = ($existing -and $existing.State -eq 'Disabled')

    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Force `
        -Description ("Auto-commits and pushes {0} on branch {1}, attributed to that branch's owner." -f $worktree, $branch) | Out-Null

    if ($wasDisabled) {
        Disable-ScheduledTask -TaskName $taskName | Out-Null
        Write-Host "Registered '$taskName' -> $worktree (branch $branch) - left disabled."
        continue
    }

    Write-Host "Registered '$taskName' -> $worktree (branch $branch)."

    if ($Start) {
        Start-ScheduledTask -TaskName $taskName
        Write-Host "  started."
    }
}

if (-not $Start) {
    Write-Host ''
    Write-Host "Start them now with:  powershell -ExecutionPolicy Bypass -File tools\autostart.ps1 -Start"
}
