<#
.SYNOPSIS
    Auto-commits and pushes every change in the Automated repo, attributing
    each commit to the contributor whose folder it came from.

.DESCRIPTION
    Changes under Anon/    are committed as the Anon identity.
    Changes under Dripper/ are committed as the Dripper identity.
    Anything else (repo root, tools/) is committed as the shared identity.

    A single save touching both folders produces two commits, one per author,
    so `git log --author` and the GitHub contributor graph stay accurate.

    Identities are read from tools/identities.json.

.PARAMETER Once
    Run a single commit+push pass and exit, instead of watching.

.PARAMETER DebounceSeconds
    How long to wait for changes to settle before committing.

.PARAMETER NoPush
    Commit locally but never push.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\sync.ps1
    powershell -ExecutionPolicy Bypass -File tools\sync.ps1 -Once
#>
[CmdletBinding()]
param(
    [switch] $Once,
    [double] $DebounceSeconds = 2,
    [switch] $NoPush
)

# 'Continue', not 'Stop': in Windows PowerShell 5.1 a native command writing to
# stderr under -ErrorAction Stop raises NativeCommandError even on exit code 0,
# and git narrates constantly. Exit codes are checked explicitly instead.
$ErrorActionPreference = 'Continue'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogFile  = Join-Path $PSScriptRoot 'sync.log'
$IdFile   = Join-Path $PSScriptRoot 'identities.json'

$script:GitExitCode = 0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
function Write-Log {
    param([string] $Message, [string] $Level = 'INFO')
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -Encoding utf8 } catch { }
}

# ---------------------------------------------------------------------------
# Git helpers. Never throw; callers check $script:GitExitCode.
# ---------------------------------------------------------------------------
function Invoke-Git {
    param([string[]] $GitArgs)
    $out = & git -C $RepoRoot @GitArgs 2>&1 | ForEach-Object { "$_" }
    $script:GitExitCode = $LASTEXITCODE
    return $out
}

# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------
function Get-Identities {
    $json = Get-Content -Path $IdFile -Raw | ConvertFrom-Json
    $map = [ordered]@{}
    foreach ($prop in $json.folders.PSObject.Properties) {
        $map[$prop.Name] = $prop.Value
    }
    return @{ Folders = $map; Shared = $json.shared }
}

# ---------------------------------------------------------------------------
# Commit subject built from staged name-status output.
# ---------------------------------------------------------------------------
function New-CommitMessage {
    param([string] $Author, [string[]] $NameStatus)

    $added = 0; $modified = 0; $deleted = 0; $renamed = 0
    $paths = @()

    foreach ($line in $NameStatus) {
        if (-not $line) { continue }
        $parts = $line -split "`t"
        $code  = $parts[0].Substring(0, 1)
        $paths += $parts[-1]
        switch ($code) {
            'A'     { $added++ }
            'M'     { $modified++ }
            'D'     { $deleted++ }
            'R'     { $renamed++ }
            default { $modified++ }
        }
    }

    if ($paths.Count -eq 1) {
        if ($added)        { $verb = 'add' }
        elseif ($deleted)  { $verb = 'delete' }
        elseif ($renamed)  { $verb = 'rename' }
        else               { $verb = 'update' }
        return '{0}: {1} {2}' -f $Author, $verb, $paths[0]
    }

    $bits = @()
    if ($added)    { $bits += "$added added" }
    if ($modified) { $bits += "$modified modified" }
    if ($deleted)  { $bits += "$deleted deleted" }
    if ($renamed)  { $bits += "$renamed renamed" }
    return '{0}: {1} files ({2})' -f $Author, $paths.Count, ($bits -join ', ')
}

# ---------------------------------------------------------------------------
# Stage one pathspec group and commit it under the matching author.
# Returns $true when a commit was created.
# ---------------------------------------------------------------------------
function Invoke-GroupCommit {
    param([string] $Author, [string] $Email, [string[]] $PathSpec)

    # Start from a clean index so each group commits only its own paths.
    Invoke-Git @('reset', '-q') | Out-Null
    Invoke-Git (@('add', '-A', '--') + $PathSpec) | Out-Null

    $status = @(Invoke-Git @('diff', '--cached', '--name-status')) |
              Where-Object { $_ -ne '' }
    if ($status.Count -eq 0) { return $false }

    $message = New-CommitMessage -Author $Author -NameStatus $status

    $env:GIT_AUTHOR_NAME     = $Author
    $env:GIT_AUTHOR_EMAIL    = $Email
    $env:GIT_COMMITTER_NAME  = $Author
    $env:GIT_COMMITTER_EMAIL = $Email
    try {
        $out = Invoke-Git @('commit', '-q', '-m', $message)
        if ($script:GitExitCode -ne 0) {
            Write-Log "commit failed for ${Author}: $out" 'WARN'
            return $false
        }
    }
    finally {
        Remove-Item Env:GIT_AUTHOR_NAME, Env:GIT_AUTHOR_EMAIL, Env:GIT_COMMITTER_NAME, Env:GIT_COMMITTER_EMAIL -ErrorAction SilentlyContinue
    }

    Write-Log $message
    return $true
}

# ---------------------------------------------------------------------------
# Commit every pending change, split by contributor folder.
# ---------------------------------------------------------------------------
function Invoke-CommitAll {
    $ids  = Get-Identities
    $made = $false

    foreach ($folder in @($ids.Folders.Keys)) {
        $id = $ids.Folders[$folder]
        if (Invoke-GroupCommit -Author $id.name -Email $id.email -PathSpec @($folder)) {
            $made = $true
        }
    }

    # Everything outside the contributor folders.
    $rest = @('.') + (@($ids.Folders.Keys) | ForEach-Object { ":(exclude)$_" })
    if (Invoke-GroupCommit -Author $ids.Shared.name -Email $ids.Shared.email -PathSpec $rest) {
        $made = $true
    }

    return $made
}

# ---------------------------------------------------------------------------
# Pull (rebase) then push.
# ---------------------------------------------------------------------------
function Test-HasUnpushed {
    $count = Invoke-Git @('rev-list', '--count', '@{u}..HEAD') | Select-Object -First 1
    if ($script:GitExitCode -ne 0) { return $true }   # no upstream yet
    return ([int] $count -gt 0)
}

function Invoke-Push {
    if ($NoPush) { return }

    $remotes = @(Invoke-Git @('remote')) | Where-Object { $_ -ne '' }
    if ($remotes -notcontains 'origin') {
        Write-Log 'no origin remote configured - commits are local only' 'WARN'
        return
    }

    $branch = Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD') | Select-Object -First 1

    $out = Invoke-Git @('pull', '--rebase', '--autostash', '-q', 'origin', $branch)
    if ($script:GitExitCode -ne 0) {
        Write-Log "pull --rebase failed, push skipped - resolve by hand: $out" 'ERROR'
        return
    }

    $out = Invoke-Git @('push', '-q', 'origin', $branch)
    if ($script:GitExitCode -ne 0) {
        Write-Log "push failed, will retry on next change: $out" 'ERROR'
        return
    }

    Write-Log "pushed $branch to origin"
}

function Invoke-Sync {
    try {
        $committed = Invoke-CommitAll
        if ($committed -or (Test-HasUnpushed)) { Invoke-Push }
    }
    catch {
        Write-Log "sync error: $_" 'ERROR'
    }
}

# ---------------------------------------------------------------------------
# Should a filesystem event trigger a sync?
# ---------------------------------------------------------------------------
function Test-RelevantPath {
    param([string] $Path)
    if (-not $Path) { return $false }

    $rel = $Path.Substring([Math]::Min($RepoRoot.Length, $Path.Length)).TrimStart('\', '/')
    if ($rel -match '^\.git($|[\\/])')            { return $false }  # our own commits
    if ($rel -match '^Anon[\\/]Dripper($|[\\/])') { return $false }  # junction
    if ($rel -match '^Dripper[\\/]Anon($|[\\/])') { return $false }  # junction
    if ($rel -match '^tools[\\/]sync\.log$')      { return $false }  # our own log
    return $true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $RepoRoot '.git'))) {
    Write-Log "no git repo at $RepoRoot" 'ERROR'
    exit 1
}

if ($Once) {
    Write-Log 'one-shot sync'
    Invoke-Sync
    exit 0
}

# One watcher per repo.
$mutexName = 'Global\AutomatedRepoSync_' + ($RepoRoot -replace '[\\/:]', '_')
$mutex = New-Object System.Threading.Mutex($false, $mutexName)
if (-not $mutex.WaitOne(0)) {
    Write-Log 'another watcher is already running - exiting' 'WARN'
    exit 0
}

Write-Log "watching $RepoRoot (debounce ${DebounceSeconds}s)"

$fsw = New-Object System.IO.FileSystemWatcher
$fsw.Path = $RepoRoot
$fsw.IncludeSubdirectories = $true
$fsw.InternalBufferSize = 65536
$fsw.NotifyFilter = [System.IO.NotifyFilters]::FileName -bor
                    [System.IO.NotifyFilters]::DirectoryName -bor
                    [System.IO.NotifyFilters]::LastWrite

foreach ($name in @('Created', 'Changed', 'Deleted', 'Renamed')) {
    Register-ObjectEvent -InputObject $fsw -EventName $name -SourceIdentifier "Fsw$name" | Out-Null
}
$fsw.EnableRaisingEvents = $true

# Catch anything edited while the watcher was down.
Invoke-Sync

$pending    = $false
$lastChange = Get-Date

try {
    while ($true) {
        $ev = Wait-Event -Timeout 1
        if ($ev) {
            foreach ($queued in @(Get-Event)) {
                if ($queued.SourceEventArgs -and
                    (Test-RelevantPath $queued.SourceEventArgs.FullPath)) {
                    $pending    = $true
                    $lastChange = Get-Date
                }
                Remove-Event -EventIdentifier $queued.EventIdentifier
            }
        }

        if ($pending -and ((Get-Date) - $lastChange).TotalSeconds -ge $DebounceSeconds) {
            $pending = $false
            Invoke-Sync
        }
    }
}
finally {
    $fsw.EnableRaisingEvents = $false
    foreach ($name in @('Created', 'Changed', 'Deleted', 'Renamed')) {
        Unregister-Event -SourceIdentifier "Fsw$name" -ErrorAction SilentlyContinue
    }
    $fsw.Dispose()
    $mutex.ReleaseMutex()
    Write-Log 'watcher stopped'
}
