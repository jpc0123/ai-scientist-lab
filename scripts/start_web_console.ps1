# 一键启动 Scientist Lab Web Console（本机开发）
# 用法：在 scientist-lab 目录执行
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_web_console.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$VenvLab = Join-Path $Root ".venv\Scripts\scientist-lab.exe"
$WebDir = Join-Path $Root "web"

if (-not (Test-Path $VenvPython)) {
  Write-Host "找不到 .venv，请先在 scientist-lab 下创建虚拟环境。" -ForegroundColor Red
  exit 1
}

Write-Host "启动后端: http://127.0.0.1:8787" -ForegroundColor Cyan
Start-Process -FilePath $VenvLab -ArgumentList @("serve", "--host", "127.0.0.1", "--port", "8787") -WorkingDirectory $Root

Start-Sleep -Seconds 2

Write-Host "启动前端: http://127.0.0.1:5173" -ForegroundColor Cyan
Start-Process -FilePath "npm" -ArgumentList @("run", "dev") -WorkingDirectory $WebDir

Write-Host ""
Write-Host "浏览器打开: http://127.0.0.1:5173" -ForegroundColor Green
Write-Host "若页面提示未连接，等几秒后刷新；或看「使用指南」。" -ForegroundColor Yellow
