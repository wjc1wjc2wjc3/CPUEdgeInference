# CPUEdgeInference Windows 部署：注册为开机自启的后台任务
# 无需额外软件（用系统自带的“计划任务”），但需**以管理员身份**运行 PowerShell。
#
# 安装：  powershell -ExecutionPolicy Bypass -File deploy\windows\install-service.ps1
# 自定义： ...\install-service.ps1 -Port 8080 -Backend llama-cpp -Threads 4
# 卸载：  ...\install-service.ps1 -Uninstall
param(
    [string]$InstallDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$Port = "8080",
    [string]$Host_ = "127.0.0.1",
    [string]$Backend = "mock",
    [string]$Threads = "0",
    [string]$PreferQuant = "auto",
    [string]$ModelDir = "",
    [switch]$Uninstall
)

$taskName = "CPUEdgeInference"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "[ok] 已卸载计划任务 $taskName"
    exit 0
}

if ([string]::IsNullOrEmpty($ModelDir)) { $ModelDir = Join-Path $InstallDir "models" }
if (-not (Test-Path $ModelDir)) { New-Item -ItemType Directory -Path $ModelDir | Out-Null }

# 优先用虚拟环境里的 Python
$venvPy = Join-Path $InstallDir ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $py = $venvPy } else { $py = "py.exe" }

# 注意：全局选项必须写在子命令 serve 之前
$arg = "-m edgeinfer.cli --model-dir `"$ModelDir`" --backend $Backend " +
       "--threads $Threads --prefer-quant $PreferQuant " +
       "serve --host $Host_ --port $Port"

$action   = New-ScheduledTaskAction -Execute $py -Argument $arg -WorkingDirectory $InstallDir
$trigger  = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "CPUEdgeInference local inference service" -Force | Out-Null

Write-Host "[ok] 已注册计划任务 $taskName"
Write-Host "     Python  : $py"
Write-Host "     工作目录 : $InstallDir"
Write-Host "     模型目录 : $ModelDir"
Write-Host "     服务地址 : http://${Host_}:$Port/v1"
Write-Host "     backend  : $Backend (threads=$Threads, quant=$PreferQuant)"
Write-Host ""
Write-Host "健康检查： curl http://${Host_}:$Port/health"
Write-Host "手动启动： Start-ScheduledTask -TaskName $taskName"
if ($Backend -eq "llama-cpp") {
    Write-Host ""
    Write-Host "提醒：真实推理需先安装 llama-cpp-python 并放入 GGUF 权重："
    Write-Host "      $py -m pip install llama-cpp-python"
}
