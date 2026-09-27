@echo off
rem CPUEdgeInference 服务启动（Windows）
rem 可用环境变量：MODEL_DIR / HOST / PORT / BACKEND / THREADS / MEMORY_BUDGET_MB / PREFER_QUANT
setlocal
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=py -3"
)

if "%MODEL_DIR%"=="" set "MODEL_DIR=.\models"
if "%HOST%"=="" set "HOST=127.0.0.1"
if "%PORT%"=="" set "PORT=8080"
if "%BACKEND%"=="" set "BACKEND=mock"
if "%THREADS%"=="" set "THREADS=0"
if "%MEMORY_BUDGET_MB%"=="" set "MEMORY_BUDGET_MB=0"
if "%PREFER_QUANT%"=="" set "PREFER_QUANT=auto"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo 启动 CPUEdgeInference
echo   backend=%BACKEND%  model-dir=%MODEL_DIR%  http://%HOST%:%PORT%
echo.

rem 注意：全局选项必须写在子命令 serve 之前
%PY% -m edgeinfer.cli --model-dir "%MODEL_DIR%" --backend "%BACKEND%" --threads "%THREADS%" --memory-budget-mb "%MEMORY_BUDGET_MB%" --prefer-quant "%PREFER_QUANT%" serve --host "%HOST%" --port "%PORT%"
endlocal
