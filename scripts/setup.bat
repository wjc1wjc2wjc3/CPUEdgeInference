@echo off
rem CPUEdgeInference 环境准备（Windows）
rem   scripts\setup.bat                          仅创建虚拟环境（核心零依赖）
rem   set INSTALL_EXTRAS=1 ^&^& scripts\setup.bat 额外安装 llama-cpp-python
rem   set NO_VENV=1 ^&^& scripts\setup.bat        不建虚拟环境
setlocal
cd /d "%~dp0.."

set "PY=py -3"
where py >nul 2>nul
if errorlevel 1 set "PY=python"

%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)"
if errorlevel 1 (
  echo [x] 需要 Python 3.9+
  exit /b 1
)

if "%NO_VENV%"=="1" goto :skipvenv
echo [1/3] 创建虚拟环境 .venv
%PY% -m venv .venv
if errorlevel 1 (
  echo [x] 创建虚拟环境失败
  exit /b 1
)
set "PY=.venv\Scripts\python.exe"
goto :aftervenv
:skipvenv
echo [1/3] 跳过虚拟环境（NO_VENV=1）
:aftervenv

echo [2/3] 升级 pip
%PY% -m pip install --upgrade pip -q

if "%INSTALL_EXTRAS%"=="1" (
  echo [3/3] 安装 llama-cpp-python（真实 CPU 推理）
  %PY% -m pip install llama-cpp-python
  if errorlevel 1 (
    echo [!] 安装失败：llama-cpp-python 在 Windows 上可能需要 Visual C++ 生成工具
    echo     可尝试： %PY% -m pip install llama-cpp-python --force-reinstall --no-cache-dir
    exit /b 1
  )
) else (
  echo [3/3] 跳过可选依赖（核心零依赖，mock 后端可直接跑通链路）
)

if not exist "models" mkdir models

echo.
echo 完成。接下来：
echo   scripts\start.bat
echo   set BACKEND=llama-cpp ^&^& scripts\start.bat
echo.
echo 把 GGUF 权重放进 models\ 即可（本项目不联网下载）。
endlocal
