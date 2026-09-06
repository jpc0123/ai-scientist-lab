# 一键安装 Scientist Lab 工作台（Windows）
# 用法（在 scientist-lab 目录）:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_workbench_common.ps1"
$Root = Get-LabRoot
Set-Location $Root

Write-Host "=== Scientist Lab 安装 ===" -ForegroundColor Cyan
Write-Host "目录: $Root"

# --- Python ---
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
  Write-Host "未找到 python。请先安装 Python 3.11+ 并加入 PATH。" -ForegroundColor Red
  exit 1
}
$verText = & python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
$parts = $verText.Split(".")
$major = [int]$parts[0]; $minor = [int]$parts[1]
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
  Write-Host "需要 Python 3.11+，当前: $verText" -ForegroundColor Red
  exit 1
}
Write-Host "Python $verText OK" -ForegroundColor Green

# --- venv ---
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  Write-Host "创建虚拟环境 .venv ..." -ForegroundColor Cyan
  & python -m venv .venv
}
Write-Host "升级 pip / 安装后端依赖 ..." -ForegroundColor Cyan
& $venvPython -m pip install -U pip setuptools wheel
if (Test-Path (Join-Path $Root "requirements.txt")) {
  & $venvPython -m pip install -r requirements.txt
}
& $venvPython -m pip install -e .

# --- Node / frontend ---
$npm = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npm) {
  Write-Host "未找到 npm。请安装 Node.js 18+（前端 React/Vite 需要）。" -ForegroundColor Red
  exit 1
}
$webDir = Join-Path $Root "web"
Write-Host "安装前端依赖 (npm install) ..." -ForegroundColor Cyan
Push-Location $webDir
try {
  npm install
} finally {
  Pop-Location
}

# --- config ---
$cfg = Join-Path $Root "config\scientist-lab.yaml"
if (-not (Test-Path $cfg)) {
  Write-Host "缺少 config\scientist-lab.yaml，请从仓库恢复该文件。" -ForegroundColor Yellow
} else {
  Write-Host "配置文件: $cfg" -ForegroundColor Green
}

# --- init db ---
Write-Host "初始化数据库 Schema ..." -ForegroundColor Cyan
& $venvPython -c "from scientist_lab.services.experiment_service import ExperimentService; ExperimentService(); print('DB OK')"

Write-Host ""
Write-Host "安装完成。" -ForegroundColor Green
Write-Host "架构说明:" -ForegroundColor Cyan
Write-Host "  后端 API = FastAPI  (默认 http://127.0.0.1:8787)"
Write-Host "  前端 UI  = React/Vite (默认 http://127.0.0.1:5173，代理 /api)"
Write-Host ""
Write-Host "下一步启动工作台:" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1"
