# 升级依赖与前端包（会备份数据库，不会静默做不可逆迁移）
# 用法:
#   powershell -ExecutionPolicy Bypass -File .\scripts\upgrade_windows.ps1

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_workbench_common.ps1"
$Root = Get-LabRoot
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
$venvLab = Join-Path $Root ".venv\Scripts\scientist-lab.exe"
if (-not (Test-Path $venvPython)) {
  Write-Host "未找到 .venv，请先 install_windows.ps1" -ForegroundColor Red
  exit 1
}

Write-Host "=== 升级工作台 ===" -ForegroundColor Cyan

# backup DB
$db = Join-Path $Root "scientist_lab.db"
$backupDir = Join-Path $Root "runtime\backups"
if (-not (Test-Path $backupDir)) {
  New-Item -ItemType Directory -Path $backupDir | Out-Null
}
if (Test-Path $db) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $dest = Join-Path $backupDir "scientist_lab_$stamp.db"
  Copy-Item $db $dest
  Write-Host "已备份数据库 -> $dest" -ForegroundColor Green
} else {
  Write-Host "未找到 scientist_lab.db，跳过备份。" -ForegroundColor Yellow
}

Write-Host "更新 Python 包 ..." -ForegroundColor Cyan
& $venvPython -m pip install -U pip
if (Test-Path (Join-Path $Root "requirements.txt")) {
  & $venvPython -m pip install -r requirements.txt
}
& $venvPython -m pip install -e .

Write-Host "更新前端依赖 ..." -ForegroundColor Cyan
Push-Location (Join-Path $Root "web")
try { npm install } finally { Pop-Location }

Write-Host "校验 Schema / Doctor ..." -ForegroundColor Cyan
& $venvPython -c "from scientist_lab.services.experiment_service import ExperimentService; ExperimentService(); print('schema ok')"
& $venvLab system-doctor

Write-Host ""
Write-Host "升级完成。如需启动:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1"
Write-Host "注意: 本脚本不会静默执行不可逆 DB 迁移；有迁移时请先确认备份。" -ForegroundColor DarkYellow
