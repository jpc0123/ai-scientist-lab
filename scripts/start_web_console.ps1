# 兼容旧入口：转发到 start_workbench.ps1
$ErrorActionPreference = "Stop"
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_workbench.ps1")
