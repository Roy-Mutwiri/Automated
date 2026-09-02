<#
.SYNOPSIS
    Registers (or removes) the scheduled task that starts the repo watcher at
    logon.

.PARAMETER Remove
    Unregister the task instead of creating it.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\autostart.ps1
    powershell -ExecutionPolicy Bypass -File tools\autostart.ps1 -Remove
#>
[CmdletBinding()]
param([switch] $Remove)

$ErrorActionPreference = 'Stop'

$TaskName = 'AutomatedRepoSync'
$SyncPath = Join-Path $PSScriptRoot 'sync.ps1'

if ($Remove) {
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Host "Removed scheduled task '$TaskName'."
    }
    catch {
        Write-Host "No scheduled task '$TaskName' to remove."
    }
    return
}

if (-not (Test-Path $SyncPath)) { throw "sync.ps1 not found at $SyncPath" }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
    '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $SyncPath
)

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

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force `
    -Description 'Auto-commits and pushes changes in the Automated repo, attributed per contributor folder.' | Out-Null

Write-Host "Registered scheduled task '$TaskName' (starts at logon)."
Write-Host "Start it now with:  Start-ScheduledTask -TaskName $TaskName"
