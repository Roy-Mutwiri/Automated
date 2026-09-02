<#
.SYNOPSIS
    Auto-commits and pushes changes in one worktree, attributing each commit to
    the identity that owns the current BRANCH.

.DESCRIPTION
    Ownership is determined by branch, not by file path.

    Folder attribution was structurally unable to separate the Camera and
    Movement terminals, because both legitimately work inside Anon/. Two
    terminals saving at the same moment produced one commit containing both
    their work, and neither could rewrite it afterwards - this watcher
    re-committed any `git reset` within seconds and then rebased it away.

    So: if the current branch is listed in identities.json `branches`, the whole
    worktree is committed as that identity in a single commit and file paths are
    never consulted. Each terminal runs in its own worktree on its own branch,
    and only its own branch is pushed.

    The one surviving path-based rule is the fallback for branches with no
    mapping - in practice `main`, where Anon/ and Dripper/ are two separate
    projects and folder splitting is the correct attribution. Collapsing that
    would lose Dripper's contributor history, which is a different problem from
    the one branch ownership solves.

    A terminal that wants a specific commit message writes it to `.sync-message`
    in its worktree; this script uses it verbatim and deletes it. Otherwise the
    subject is generated from the staged name-status, as before.

.PARAMETER Worktree
    Which worktree to watch and commit. Defaults to the repo containing this
    script. Each terminal gets its own watcher pointed at its own worktree; a
    watcher never touches a worktree other than this one.

.PARAMETER Once
    Run a single commit+push pass and exit, instead of watching.

.PARAMETER DebounceSeconds
    How long to wait for changes to settle before committing.

.PARAMETER NoPush
    Commit locally but never push.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\sync.ps1
    powershell -ExecutionPolicy Bypass -File tools\sync.ps1 -Once -NoPush
    powershell -ExecutionPolicy Bypass -File tools\sync.ps1 -Worktree C:\Users\me\Automated-camera
#>
[CmdletBinding()]
param(
    [switch] $Once,
    [double] $DebounceSeconds = 2,
    [switch] $NoPush,
    [string] $Worktree
)

# 'Continue', not 'Stop': in Windows PowerShell 5.1 a native command writing to
# stderr under -ErrorAction Stop raises NativeCommandError even on exit code 0,
# and git narrates constantly. Exit codes are checked explicitly instead.
$ErrorActionPreference = 'Continue'

# The worktree this watcher owns. Every git call below is scoped to it, so two
# watchers on two worktrees cannot commit each other's work.
if ($Worktree) {
    $RepoRoot = (Resolve-Path -LiteralPath $Worktree).Path
} else {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
$LogFile  = Join-Path $PSScriptRoot ('sync-{0}.log' -f (Split-Path -Leaf $RepoRoot))
$IdFile   = Join-Path $PSScriptRoot 'identities.json'
$MsgFile  = Join-Path $RepoRoot '.sync-message'

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
    $branches = [ordered]@{}
    if ($json.branches) {
        foreach ($prop in $json.branches.PSObject.Properties) {
            $branches[$prop.Name] = $prop.Value
        }
    }
    return @{ Folders = $map; Branches = $branches; Shared = $json.shared }
}

# The branch this worktree is on, or $null when detached. A detached HEAD has no
# owner and nothing sensible to push to, so syncing stops rather than guesses.
function Get-CurrentBranch {
    $branch = Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD') | Select-Object -First 1
    if ($script:GitExitCode -ne 0 -or -not $branch -or $branch -eq 'HEAD') { return $null }
    return $branch.Trim()
}

# A commit message the terminal asked for, if it left one. Read once and
# removed, so a stale file cannot re-label a later unrelated commit.
function Read-IntendedMessage {
    if (-not (Test-Path -LiteralPath $MsgFile)) { return $null }
    try {
        $text = (Get-Content -LiteralPath $MsgFile -Raw).Trim()
        Remove-Item -LiteralPath $MsgFile -Force -ErrorAction SilentlyContinue
        if ($text) { return $text }
    } catch { }
    return $null
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
    param([string] $Author, [string] $Email, [string[]] $PathSpec, [string] $Message)

    # Start from a clean index so each group commits only its own paths.
    Invoke-Git @('reset', '-q') | Out-Null
    Invoke-Git (@('add', '-A', '--') + $PathSpec) | Out-Null

    $status = @(Invoke-Git @('diff', '--cached', '--name-status')) |
              Where-Object { $_ -ne '' }
    if ($status.Count -eq 0) { return $false }

    if ($Message) { $message = $Message }
    else          { $message = New-CommitMessage -Author $Author -NameStatus $status }

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
    param([string] $Branch)

    $ids  = Get-Identities
    $made = $false

    # ---- Branch ownership. The whole worktree, one commit, no path rules. ----
    if ($Branch -and $ids.Branches.Contains($Branch)) {
        $id = $ids.Branches[$Branch]
        $intended = Read-IntendedMessage
        if (Invoke-GroupCommit -Author $id.name -Email $id.email `
                               -PathSpec @('.') -Message $intended) {
            $made = $true
        }
        return $made
    }

    # ---- Fallback: no branch mapping, so split by project folder. ----
    # This is the only remaining path-based classification, and it exists for
    # `main`, where Anon/ and Dripper/ really are separate projects. A terminal
    # branch never reaches here.
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
    param([string] $Branch)

    if ($NoPush) { return }
    if (-not $Branch) { return }

    $remotes = @(Invoke-Git @('remote')) | Where-Object { $_ -ne '' }
    if ($remotes -notcontains 'origin') {
        Write-Log 'no origin remote configured - commits are local only' 'WARN'
        return
    }

    # Only this worktree's own branch is ever pulled or pushed. A watcher must
    # not move a branch another terminal is sitting on.
    #
    # A branch origin has never seen has nothing to pull: `pull origin <branch>`
    # fails with "couldn't find remote ref", which would abort the push and
    # leave a new terminal branch permanently local. Publish it instead, and
    # let -u set the upstream so every later pass takes the pull path.
    Invoke-Git @('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}') | Out-Null
    $hasUpstream = ($script:GitExitCode -eq 0)

    if ($hasUpstream) {
        $out = Invoke-Git @('pull', '--rebase', '--autostash', '-q', 'origin', $Branch)
        if ($script:GitExitCode -ne 0) {
            Write-Log "pull --rebase failed, push skipped - resolve by hand: $out" 'ERROR'
            return
        }
        $out = Invoke-Git @('push', '-q', 'origin', $Branch)
    }
    else {
        Write-Log "publishing $Branch to origin for the first time"
        $out = Invoke-Git @('push', '-q', '-u', 'origin', $Branch)
    }

    if ($script:GitExitCode -ne 0) {
        Write-Log "push failed, will retry on next change: $out" 'ERROR'
        return
    }

    Write-Log "pushed $Branch to origin"
}

function Invoke-Sync {
    try {
        $branch = Get-CurrentBranch
        if (-not $branch) {
            Write-Log 'detached HEAD - no branch owns this worktree, sync skipped' 'WARN'
            return
        }
        $committed = Invoke-CommitAll -Branch $branch
        if ($committed -or (Test-HasUnpushed)) { Invoke-Push -Branch $branch }
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
    if ($rel -match '^tools[\\/]sync.*\.log$')    { return $false }  # our own log
    if ($rel -match '^\.sync-message$')           { return $false }  # consumed below
    if ($rel -match '^\.claude[\\/]worktrees[\\/]') { return $false }  # another worktree
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
