<#
.SYNOPSIS
  Start the API, open a tunnel to it, and republish the Vercel page pointed at that tunnel.

.DESCRIPTION
  The models run on this machine; the page is static and lives on Vercel. That split works,
  but a quick tunnel gets a NEW random hostname every time it starts, so every reboot used
  to mean copying a fresh URL into the page by hand or sharing a ?api= link that died with
  the session.

  This closes that loop. It starts the server, waits for it to actually answer, opens the
  tunnel, waits for that to actually answer, and only then rebuilds the page with the new
  address baked in as window.__API_BASE__ and redeploys it. After it finishes,

      https://ai-powered-radiology-hub.vercel.app

  works on its own - no query string, no copied link - for as long as this machine stays up.

  Each step waits on a real response rather than a sleep. A tunnel that is "up" before
  cloudflared has registered its edge connection answers 502, and a page published against
  that address is published against a hostname that does not resolve yet.

.PARAMETER SkipDeploy
  Do everything except touch Vercel. Prints the tunnel URL for a one-off ?api= link.

.PARAMETER Install
  Register this script to run at logon via Task Scheduler, then exit.

.PARAMETER Uninstall
  Remove that scheduled task, then exit.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\serve-and-publish.ps1
  powershell -ExecutionPolicy Bypass -File scripts\serve-and-publish.ps1 -Install
#>
[CmdletBinding()]
param(
    [switch]$SkipDeploy,
    [switch]$Install,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
$Root      = Split-Path -Parent $PSScriptRoot
$Python    = Join-Path $Root 'api\venv\Scripts\python.exe'
$Cloudflared = Join-Path $Root 'tools\cloudflared.exe'
$LogDir    = Join-Path $Root '.run'
$ApiLog    = Join-Path $LogDir 'api.log'
$TunnelLog = Join-Path $LogDir 'tunnel.log'
$TaskName  = 'RadiologyHub-ServeAndPublish'
$ApiUrl    = 'http://127.0.0.1:8000'

function Say($msg) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $msg" }

# --- install / uninstall -----------------------------------------------------------------
if ($Install) {
    # Logon rather than boot: the task needs the user's Vercel and cloudflared credentials,
    # and a boot-time SYSTEM task has neither.
    $action  = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`"" `
        -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    # Start slightly late: at logon the network stack is often not routable yet, and the
    # first thing this does is wait on a remote host.
    $trigger.Delay = 'PT45S'
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Description 'Serve the Radiology Hub API and publish its tunnel URL' -Force | Out-Null
    Say "installed scheduled task '$TaskName' (runs 45s after logon)"
    Say "remove it with: -Uninstall"
    return
}
if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Say "removed scheduled task '$TaskName'"
    return
}

New-Item -ItemType Directory -Force $LogDir | Out-Null

# --- 0. cloudflared ----------------------------------------------------------------------
if (-not (Test-Path $Cloudflared)) {
    Say 'cloudflared not present - downloading (55 MB)'
    New-Item -ItemType Directory -Force (Split-Path $Cloudflared) | Out-Null
    Invoke-WebRequest -UseBasicParsing -OutFile $Cloudflared `
        'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
}

# --- 1. the API --------------------------------------------------------------------------
function Test-Api {
    try { return (Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 5).status -eq 'ok' }
    catch { return $false }
}

if (Test-Api) {
    Say 'api already answering on 8000 - leaving it alone'
} else {
    Say 'starting the api'
    Start-Process -FilePath $Python `
        -ArgumentList '-m','uvicorn','main:app','--host','127.0.0.1','--port','8000' `
        -WorkingDirectory (Join-Path $Root 'api') `
        -WindowStyle Hidden `
        -RedirectStandardOutput $ApiLog -RedirectStandardError "$ApiLog.err"
    # Loading 14 models off disk takes well under a minute cold, longer on a slow disk.
    $deadline = (Get-Date).AddMinutes(4)
    while (-not (Test-Api)) {
        if ((Get-Date) -gt $deadline) { throw "api did not come up in 4 minutes - see $ApiLog.err" }
        Start-Sleep -Seconds 3
    }
}
$health = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 10
$ready  = ($health.models.PSObject.Properties | Where-Object { $_.Value }).Count
Say "api up - $ready models on $($health.device)"

# --- 2. the tunnel -----------------------------------------------------------------------
# A quick tunnel is handed a random hostname, and that hostname is not always usable. Seen
# on this machine: cloudflared registered its edge connection and printed a URL that never
# got an A record - IPv6 only, on an IPv4-only network, so unreachable from here and from
# anyone else without IPv6. Nothing in cloudflared's output says so; it looks like success.
#
# There is no fixing a bad name, only asking for another one, so this takes up to three and
# only accepts one it has actually fetched /health through. The alternative - trusting the
# printed URL - publishes a page pointing at a host that does not resolve.
function Start-Tunnel {
    Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
    Remove-Item $TunnelLog -ErrorAction SilentlyContinue
    Start-Process -FilePath $Cloudflared `
        -ArgumentList 'tunnel','--url',$ApiUrl,'--no-autoupdate' `
        -WindowStyle Hidden -RedirectStandardError $TunnelLog -RedirectStandardOutput "$TunnelLog.out"

    $url = $null
    $deadline = (Get-Date).AddSeconds(60)
    while (-not $url -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if (Test-Path $TunnelLog) {
            $m = [regex]::Match((Get-Content $TunnelLog -Raw -ErrorAction SilentlyContinue),
                                'https://[a-z0-9-]+\.trycloudflare\.com')
            if ($m.Success) { $url = $m.Value }
        }
    }
    return $url
}

$tunnel = $null
foreach ($try in 1..3) {
    Say "opening the tunnel (attempt $try of 3)"
    $candidate = Start-Tunnel
    if (-not $candidate) { Say '    no URL printed'; continue }
    Say "    got $candidate - checking it actually serves"

    # 75s: a usable name answers within a few seconds of registering. A name that is going
    # to stay broken shows no A record at all, and waiting longer does not create one.
    $ok = $false
    $deadline = (Get-Date).AddSeconds(75)
    while ((Get-Date) -lt $deadline) {
        try {
            if ((Invoke-RestMethod -Uri "$candidate/health" -TimeoutSec 15).status -eq 'ok') { $ok = $true; break }
        } catch { }
        Start-Sleep -Seconds 5
    }
    if ($ok) { $tunnel = $candidate; break }
    Say '    that hostname never served - discarding it and asking for another'
}
if (-not $tunnel) {
    Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
    throw "three tunnels in a row were unreachable - see $TunnelLog. The api itself is fine on $ApiUrl."
}
Say "tunnel is serving the api: $tunnel"

if ($SkipDeploy) {
    Say 'skipping deploy as asked'
    Say "use: https://ai-powered-radiology-hub.vercel.app/?api=$tunnel"
    return
}

# --- 3. republish the page with this address baked in ------------------------------------
Say 'publishing the page against it'
$env:API_BASE = $tunnel
Push-Location $Root
try {
    $out = & vercel deploy --prod --yes 2>&1
    $live = ([regex]::Match(($out -join "`n"), 'https://[a-z0-9.-]*vercel\.app')).Value
    if (-not $live) { $out | ForEach-Object { Write-Host "    $_" }; throw 'vercel did not return a URL' }
} finally {
    Pop-Location
    Remove-Item Env:\API_BASE -ErrorAction SilentlyContinue
}

Say 'done'
Write-Host ''
Write-Host '  https://ai-powered-radiology-hub.vercel.app' -ForegroundColor Green
Write-Host '  works on its own now - no ?api= needed, while this machine is up.'
Write-Host ''
