# Windows 环境诊断（包装 system-doctor）
# 用法:
#   powershell -ExecutionPolicy Bypass -File .\scripts\doctor_windows.ps1

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_workbench_common.ps1"
$Root = Get-LabRoot
Set-Location $Root

$venvLab = Join-Path $Root ".venv\Scripts\scientist-lab.exe"
if (-not (Test-Path $venvLab)) {
  Write-Host "未安装。请先运行 install_windows.ps1" -ForegroundColor Red
  exit 1
}

$cfg = Get-WorkbenchConfig -Root $Root
Write-Host "=== System Doctor ===" -ForegroundColor Cyan
Write-Host "后端 FastAPI: $($cfg.ApiUrl)  前端 React: $($cfg.WebUrl)"
& $venvLab system-doctor
$code = $LASTEXITCODE

Write-Host ""
Write-Host "端口状态:" -ForegroundColor Cyan
$apiUp = Test-PortListening -HostName $cfg.ApiHost -Port $cfg.ApiPort
$webUp = Test-PortListening -HostName $cfg.WebHost -Port $cfg.WebPort
Write-Host ("  API  {0}:{1} -> {2}" -f $cfg.ApiHost, $cfg.ApiPort, ($(if ($apiUp) {"listening"} else {"down"})))
Write-Host ("  Web  {0}:{1} -> {2}" -f $cfg.WebHost, $cfg.WebPort, ($(if ($webUp) {"listening"} else {"down"})))

exit $code
