@echo off
setlocal EnableDelayedExpansion
title AssetPixelHand Universal Launcher

:: CHANGE DIRECTORY TO THE SCRIPT'S LOCATION
cd /d "%~dp0"

:: --- Configuration ---
set "VENV_DIR=.venv"
set "PYTHON_EXE=python"

:: --- Argument Parsing ---
:: [MODIFIED] We now use a flag that will be passed directly to the Python script.
set "DEBUG_FLAG="
set "REINSTALL_MODE=0"
set "DIAG_MODE=0"
set "PROFILE_MODE=0"

if /i "%1"=="debug" ( set "DEBUG_FLAG=--debug" && echo ** DEBUG MODE ACTIVATED ** && echo. )
if /i "%1"=="reinstall" ( set "REINSTALL_MODE=1" && echo ** REINSTALL MODE ACTIVATED ** && echo. )
if /i "%1"=="diag" ( set "DIAG_MODE=1" && echo ** DIAGNOSTIC MODE ACTIVATED ** && echo. )
if /i "%1"=="profile" ( set "PROFILE_MODE=1" && echo ** PROFILING MODE ACTIVATED ** && echo. )

:: --- Header ---
echo =======================================================
echo         AssetPixelHand Universal Launcher
echo =======================================================
echo This script will set up the environment and run the application.
echo.

:: --- [1/5] Project Sanity Check ---
echo [1/5] Verifying project structure...
if not exist "pyproject.toml" ( goto :error "pyproject.toml not found. Please run this script from the project root." )
if not exist "app\diagnostics.py" ( goto :error "The 'app' directory or 'diagnostics.py' is missing." )
if not exist "main.py" ( goto :error "The main entry point 'main.py' is missing from the project root." )
echo [OK] Project structure is valid.
echo.

:: --- [2/5] Virtual Environment Setup ---
if "!REINSTALL_MODE!"=="1" (
    if exist "%VENV_DIR%" (
        echo [2/5] Reinstall mode: Deleting existing virtual environment...
        rmdir /s /q "%VENV_DIR%"
        if !errorlevel! neq 0 ( goto :error "Could not delete the existing '%VENV_DIR%' directory. Check for file locks." )
    )
)
set "NEEDS_INSTALL=0"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [2/5] Creating Python virtual environment in '%VENV_DIR%'...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if !errorlevel! neq 0 ( goto :error "Failed to create the virtual environment. Is Python installed and in your PATH?" )
    set "NEEDS_INSTALL=1"
) else (
    echo [2/5] Virtual environment already exists.
)
echo Activating virtual environment...
set "PATH=%CD%\%VENV_DIR%\Scripts;%PATH%"
echo [OK] Virtual environment is active.
echo.

:: --- [3/5] Installing/Verifying Dependencies ---
set "SHOULD_INSTALL=0"
if "!NEEDS_INSTALL!"=="1" set "SHOULD_INSTALL=1"
if "!REINSTALL_MODE!"=="1" set "SHOULD_INSTALL=1"

if "!SHOULD_INSTALL!"=="1" (
    echo [3/5] Installing dependencies... This may take a few minutes.
    pip install --upgrade pip

    :: Install all dependencies from pyproject.toml
    echo --- Installing project dependencies...
    echo Installing GPU-accelerated libraries...
    pip install --upgrade ".[gpu]"
    if !errorlevel! neq 0 (
        echo.
        echo [WARNING] GPU-accelerated library installation failed. Trying CPU-only fallback...
        pip install --upgrade ".[cpu]"
        if !errorlevel! neq 0 ( goto :error "All installation attempts failed." )
    )

) else (
    echo [3/5] Dependencies appear to be installed. Skipping installation.
)
echo [OK] Dependencies are ready.
echo.

:: --- [4/5] Running Diagnostics ---
echo [4/5] Running environment diagnostics...
python -m app.diagnostics
if !errorlevel! neq 0 (
    if "!DIAG_MODE!"=="1" (
        goto :error "One or more diagnostic checks failed. Please review the output above."
    ) else (
        echo.
        echo [WARNING] One or more diagnostic checks failed. The application might not run correctly.
        pause
    )
) else (
    echo [OK] All diagnostic checks passed.
)

if "!DIAG_MODE!"=="1" (
    echo.
    echo Diagnostic run complete.
    goto :end_success
)
echo.

:: --- [5/5] Launching Application ---
echo =======================================================
echo [5/5] Starting AssetPixelHand...
if "!DEBUG_FLAG!"=="--debug" (
    echo    (Debug Mode Enabled: Log level will be set to DEBUG)
)
echo =======================================================
echo.

if "!PROFILE_MODE!"=="1" (
    echo --- RUNNING WITH CPROFILE ---
    echo The application may run slower. After you finish the scan and close the app,
    echo a file 'app_data\scan_profile.pstats' will be created.
    echo.
    :: [MODIFIED] Pass the debug flag to the python script when profiling
    python -m cProfile -o "app_data\scan_profile.pstats" main.py %DEBUG_FLAG%
) else (
    :: [MODIFIED] Pass the debug flag to the python script
    python main.py %DEBUG_FLAG%
)

if !errorlevel! neq 0 (
    goto :error "The application exited unexpectedly. Check the 'app_data\app_log.txt' file for details."
)

echo.
echo =======================================================
if "!PROFILE_MODE!"=="1" (
    echo Profiling finished. To view the results, install snakeviz:
    echo   pip install snakeviz
    echo Then run:
    echo   snakeviz app_data\scan_profile.pstats
)
echo Application finished. Thank you for using AssetPixelHand!
goto :end_success

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo [FATAL ERROR] %~1
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo.
pause
exit /b 1

:end_success
endlocal
pause