# Shared helpers for Scientist Lab Windows workbench scripts.

function Get-LabRoot {
  return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-WorkbenchConfig {
  param([string]$Root)
  $cfgPath = Join-Path $Root "config\scientist-lab.yaml"
  $apiHost = "127.0.0.1"
  $apiPort = 8787
  $webHost = "127.0.0.1"
  $webPort = 5173
  $openBrowser = $true

  if (Test-Path $cfgPath) {
    $text = Get-Content -Path $cfgPath -Raw -ErrorAction SilentlyContinue
    if ($text) {
      if ($text -match '(?m)^\s*port:\s*(\d+)\s*$') {
        # First bare port under app: block — yaml order: app.port then web.port
        $ports = [regex]::Matches($text, '(?m)^\s*port:\s*(\d+)\s*$')
        if ($ports.Count -ge 1) { $apiPort = [int]$ports[0].Groups[1].Value }
        if ($ports.Count -ge 2) { $webPort = [int]$ports[1].Groups[1].Value }
      }
      if ($text -match '(?m)open_browser:\s*(true|false)') {
        $openBrowser = ($Matches[1] -eq "true")
      }
    }
  }

  return [pscustomobject]@{
    ConfigPath  = $cfgPath
    ApiHost     = $apiHost
    ApiPort     = $apiPort
    WebHost     = $webHost
    WebPort     = $webPort
    OpenBrowser = $openBrowser
    ApiUrl      = "http://${apiHost}:${apiPort}"
    WebUrl      = "http://${webHost}:${webPort}"
  }
}

function Test-PortListening {
  param([string]$HostName, [int]$Port)
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($HostName, $Port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(400)
    if ($ok -and $client.Connected) {
      $client.Close()
      return $true
    }
    $client.Close()
    return $false
  } catch {
    return $false
  }
}

function Get-PidFile {
  param([string]$Root)
  $runtime = Join-Path $Root "runtime"
  if (-not (Test-Path $runtime)) {
    New-Item -ItemType Directory -Path $runtime | Out-Null
  }
  return (Join-Path $runtime "workbench.pids.json")
}

function Save-WorkbenchPids {
  param(
    [string]$Root,
    [Nullable[int]]$ApiPid,
    [Nullable[int]]$WebPid
  )
  $path = Get-PidFile -Root $Root
  $obj = @{
    api_pid = $ApiPid
    web_pid = $WebPid
    updated_at = (Get-Date).ToString("o")
  }
  $obj | ConvertTo-Json | Set-Content -Path $path -Encoding UTF8
  return $path
}

function Read-WorkbenchPids {
  param([string]$Root)
  $path = Get-PidFile -Root $Root
  if (-not (Test-Path $path)) { return $null }
  try {
    return (Get-Content -Path $path -Raw | ConvertFrom-Json)
  } catch {
    return $null
  }
}

function Stop-PidSafe {
  param([Nullable[int]]$ProcessId, [string]$Label)
  if (-not $ProcessId) { return }
  try {
    $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($p) {
      Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
      Write-Host "已停止 $Label (PID $ProcessId)" -ForegroundColor Yellow
    }
  } catch {
    Write-Host "无法停止 $Label (PID $ProcessId): $_" -ForegroundColor DarkYellow
  }
}

function Stop-ListenersOnPort {
  param([int]$Port)
  try {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
      if ($c.OwningProcess) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "已释放端口 $Port (PID $($c.OwningProcess))" -ForegroundColor Yellow
      }
    }
  } catch {
    # Fallback: ignore if Get-NetTCPConnection unavailable
  }
}
