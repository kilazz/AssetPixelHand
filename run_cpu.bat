@echo off
setlocal EnableDelayedExpansion
title AssetPixelHand Launcher (CPU)

:: CHANGE DIRECTORY TO THE SCRIPT'S LOCATION
cd /d "%~dp0"

:: --- Configuration ---
set "VENV_DIR=.venv-cpu"
set "ONNX_BACKEND=cpu"
set "PYTHON_EXE=python"
set "REINSTALL_MODE="
set "DIAG_MODE="
set "PROFILE_MODE="

:: --- Argument Parsing ---
for %%a in (%*) do (
    if /i "%%a"=="--reinstall" ( set "REINSTALL_MODE=1" )
    if /i "%%a"=="--diag" ( set "DIAG_MODE=1" )
    if /i "%%a"=="--profile" ( set "PROFILE_MODE=1" )
)

:: --- Header ---
echo =======================================================
echo         AssetPixelHand Launcher for [CPU]
echo =======================================================
echo.

:: --- [1/5] Project Sanity Check ---
echo [1/5] Verifying project structure...
if not exist "pyproject.toml" ( goto :error "pyproject.toml not found. Please run this script from the project root." )
echo [OK] Project structure is valid.
echo.

:: --- [2/5] Virtual Environment Setup ---
if defined REINSTALL_MODE (
    if exist "%VENV_DIR%" (
        echo [2/5] Reinstall mode: Deleting existing '%VENV_DIR%'...
        rmdir /s /q "%VENV_DIR%"
        if errorlevel 1 ( goto :error "Could not delete the '%VENV_DIR%' directory." )
    )
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [2/5] Creating Python virtual environment in '%VENV_DIR%'...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if errorlevel 1 ( goto :error "Failed to create the virtual environment." )
    set "NEEDS_INSTALL=1"
) else (
    echo [2/5] Virtual environment '%VENV_DIR%' already exists.
)

echo Activating virtual environment...
set "PATH=%CD%\%VENV_DIR%\Scripts;%PATH%"
echo [OK] Virtual environment is active.
echo.

:: --- [3/5] Installing Dependencies ---
if not defined NEEDS_INSTALL if not defined REINSTALL_MODE (
    echo [3/5] Dependencies appear to be installed. Skipping.
    goto :deps_done
)

echo [3/5] Installing dependencies for [%ONNX_BACKEND%] backend...
pip install --upgrade pip
pip install --upgrade ".[%ONNX_BACKEND%]"
if errorlevel 1 ( goto :error "Failed to install dependencies." )

:deps_done
echo [OK] Dependencies are ready.
echo.

:: --- [4/5] Running Diagnostics ---
echo [4/5] Running environment diagnostics...
python -m app.diagnostics
if errorlevel 1 (
    echo [WARNING] One or more diagnostic checks failed.
    pause
) else (
    echo [OK] All diagnostic checks passed.
)
if defined DIAG_MODE ( goto :end_success )
echo.

:: --- [5/5] Launching Application ---
echo =======================================================
echo [5/5] Starting AssetPixelHand with [%ONNX_BACKEND%]...
echo =======================================================
echo.

if defined PROFILE_MODE (
    echo [INFO] Running in profile mode. All arguments like --debug will be passed.
    python -m cProfile -o "app_data\scan_profile.pstats" main.py %*
) else (
    echo [INFO] Passing arguments to application: %*
    python main.py %*
)

if errorlevel 1 ( goto :error "Application exited unexpectedly." )
goto :end_success

:error
echo. & echo !!!!!!!!!!!!!!!!!!!! [FATAL ERROR] %~1 !!!!!!!!!!!!!!!!!!!! & echo.
pause
exit /b 1

:end_success
endlocal
echo. & echo Script finished.
pause