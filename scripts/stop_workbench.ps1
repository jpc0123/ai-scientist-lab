# Stop Scientist Lab workbench
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_workbench_common.ps1"
$Root = Get-LabRoot
$cfg = Get-WorkbenchConfig -Root $Root

Write-Host "=== Stop workbench ===" -ForegroundColor Cyan
$pids = Read-WorkbenchPids -Root $Root
if ($pids) {
  Stop-PidSafe -ProcessId $pids.api_pid -Label "API"
  Stop-PidSafe -ProcessId $pids.web_pid -Label "Web"
}

Stop-ListenersOnPort -Port $cfg.ApiPort
Stop-ListenersOnPort -Port $cfg.WebPort

$pidFile = Get-PidFile -Root $Root
if (Test-Path $pidFile) {
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Write-Host ("Stopped. API {0} / Web {1}" -f $cfg.ApiPort, $cfg.WebPort) -ForegroundColor Green
