@echo off
REM Start the auto-commit/push watcher in this window (Ctrl+C to stop).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync.ps1" %*
