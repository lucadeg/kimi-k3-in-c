@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "KIMI_PORT=8095"

echo ================================================================================
echo  [KIMI/HYDRA] Starting First-Layer Verification + Heavy Router Bridge (Port: %KIMI_PORT%)
echo ================================================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    exit /b 1
)

python "%SCRIPT_DIR%kimi_k3_service.py"
