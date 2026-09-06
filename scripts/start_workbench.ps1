# One-click start Scientist Lab workbench (API + Web)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_workbench_common.ps1"
$Root = Get-LabRoot
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
$venvLab = Join-Path $Root ".venv\Scripts\scientist-lab.exe"
$webDir = Join-Path $Root "web"

if (-not (Test-Path $venvPython) -or -not (Test-Path $venvLab)) {
  Write-Host "[ERROR] .venv not found. Run install_windows.ps1 first." -ForegroundColor Red
  Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1"
  exit 1
}

$cfg = Get-WorkbenchConfig -Root $Root
Write-Host "=== Start workbench ===" -ForegroundColor Cyan
Write-Host ("API  (FastAPI): {0}" -f $cfg.ApiUrl)
Write-Host ("Web  (React):   {0}" -f $cfg.WebUrl)
Write-Host ("Config:         {0}" -f $cfg.ConfigPath)

if (Test-PortListening -HostName $cfg.ApiHost -Port $cfg.ApiPort) {
  Write-Host ("API port {0} already listening; skip backend." -f $cfg.ApiPort) -ForegroundColor Yellow
  $apiProc = $null
} else {
  Write-Host "Starting FastAPI backend ..." -ForegroundColor Cyan
  $apiProc = Start-Process -FilePath $venvLab `
    -ArgumentList @("serve", "--host", $cfg.ApiHost, "--port", "$($cfg.ApiPort)") `
    -WorkingDirectory $Root `
    -WindowStyle Minimized `
    -PassThru
}

if (Test-PortListening -HostName $cfg.WebHost -Port $cfg.WebPort) {
  Write-Host ("Web port {0} already listening; skip frontend." -f $cfg.WebPort) -ForegroundColor Yellow
  $webProc = $null
} else {
  $npmCmd = Resolve-NpmCmd
  if (-not $npmCmd) {
    Write-Host "[ERROR] npm.cmd not found. Install Node.js and add it to PATH." -ForegroundColor Red
    exit 1
  }
  Write-Host ("Starting React/Vite frontend via {0} ..." -f $npmCmd) -ForegroundColor Cyan
  # Must use npm.cmd. Start-Process "npm" on Windows often opens the extensionless
  # npm shim in Notepad, so Vite never starts and the UI looks "unable to open".
  $webProc = Start-Process -FilePath $npmCmd `
    -ArgumentList @("run", "dev", "--", "--host", $cfg.WebHost, "--port", "$($cfg.WebPort)", "--strictPort") `
    -WorkingDirectory $webDir `
    -WindowStyle Minimized `
    -PassThru
}

$existing = Read-WorkbenchPids -Root $Root
$apiPid = if ($apiProc) { $apiProc.Id } elseif ($existing) { $existing.api_pid } else { $null }
$webPid = if ($webProc) { $webProc.Id } elseif ($existing) { $existing.web_pid } else { $null }
Save-WorkbenchPids -Root $Root -ApiPid $apiPid -WebPid $webPid | Out-Null

Write-Host "Waiting for API ..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 1
  if (Test-PortListening -HostName $cfg.ApiHost -Port $cfg.ApiPort) {
    $ready = $true
    break
  }
}
if ($ready) {
  Write-Host "API is ready." -ForegroundColor Green
} else {
  Write-Host "API not ready yet; refresh the browser later." -ForegroundColor Yellow
}

Write-Host "Waiting for Web ..." -ForegroundColor Cyan
$webReady = $false
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 1
  if (Test-PortListening -HostName $cfg.WebHost -Port $cfg.WebPort) {
    $webReady = $true
    break
  }
}
if ($webReady) {
  Write-Host "Web is ready." -ForegroundColor Green
} else {
  Write-Host ("Web not listening on {0}; check the minimized npm window." -f $cfg.WebUrl) -ForegroundColor Yellow
}

if ($cfg.OpenBrowser) {
  Start-Process $cfg.WebUrl
}

Write-Host ""
Write-Host ("Open browser: {0}" -f $cfg.WebUrl) -ForegroundColor Green
Write-Host "Frontend proxies /api to FastAPI." -ForegroundColor DarkGray
Write-Host "Stop: powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1"
