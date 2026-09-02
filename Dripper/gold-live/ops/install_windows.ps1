<#
.SYNOPSIS
    Install Gold Live as a Windows scheduled task that starts at logon.

.DESCRIPTION
    The machine described runs LIVE Studio, so there is no systemd. Task
    Scheduler is the equivalent: it starts the supervisor at logon, restarts it
    if it dies, and never stops it for running too long.

    Deliberately runs at LOGON rather than as a SYSTEM service. Screen capture
    and audio both need an interactive desktop session -- a SYSTEM service has
    no desktop to capture and no audio endpoint to play to, so it would install
    cleanly and then silently do nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ops\install_windows.ps1
    powershell -ExecutionPolicy Bypass -File ops\install_windows.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch] $Remove,
    [string] $TaskName = "GoldLive",
    [string] $ExePath,
    [string] $Arguments = "supervise"
)

$ErrorActionPreference = "Stop"

if ($Remove) {
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Host "Removed scheduled task '$TaskName'."
    } catch {
        Write-Host "No scheduled task '$TaskName' to remove."
    }
    return
}

if (-not $ExePath) {
    $candidates = @(
        (Join-Path $PSScriptRoot "..\dist\GoldLive\GoldLive.exe"),
        (Join-Path $PSScriptRoot "..\GoldLive.exe")
    )
    $ExePath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $ExePath -or -not (Test-Path $ExePath)) {
    throw "GoldLive.exe not found. Build it first, or pass -ExePath."
}
$ExePath = (Resolve-Path $ExePath).Path
$WorkDir = Split-Path -Parent $ExePath

Write-Host "  Executable : $ExePath"
Write-Host "  Arguments  : $Arguments"

$action = New-ScheduledTaskAction -Execute $ExePath -Argument $Arguments -WorkingDirectory $WorkDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# ExecutionTimeLimit of zero means "never kill this for running too long",
# which is the entire point of a 24/7 service. The default is 3 days, and
# hitting it looks exactly like an unexplained crash three days in.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force `
    -Description "Gold Live 24/7 AI broadcasting system. Starts at logon." | Out-Null

Write-Host ""
Write-Host "Registered '$TaskName' (starts at logon)."
Write-Host ""
Write-Host "  Start now    Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Stop         Stop-ScheduledTask  -TaskName $TaskName"
Write-Host "  Status       Get-ScheduledTask   -TaskName $TaskName"
Write-Host "  Uninstall    ...\install_windows.ps1 -Remove"
Write-Host ""
Write-Host "Check it is healthy:  $ExePath doctor"
Write-Host "Watch it running:     $ExePath dashboard"
