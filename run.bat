@echo off
setlocal EnableDelayedExpansion
title AssetPixelHand Universal Launcher

:: CHANGE DIRECTORY TO THE SCRIPT'S LOCATION
cd /d "%~dp0"

:: --- Configuration ---
set "VENV_DIR=.venv"
set "PYTHON_EXE=python"
set "PYTHON_ARGS="

:: --- Argument Parsing ---
:: Loop through arguments to handle them in any order
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="debug" (
    echo ** DEBUG MODE ACTIVATED **
    set "PYTHON_ARGS=!PYTHON_ARGS! --debug"
) else if /i "%~1"=="reinstall" (
    echo ** REINSTALL MODE ACTIVATED **
    set "REINSTALL_MODE=1"
) else if /i "%~1"=="diag" (
    echo ** DIAGNOSTIC MODE ACTIVATED **
    set "DIAG_MODE=1"
) else if /i "%~1"=="profile" (
    echo ** PROFILING MODE ACTIVATED **
    set "PROFILE_MODE=1"
) else (
    echo [WARNING] Unknown argument: %~1
)
shift
goto :parse_args
:args_done

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
if defined REINSTALL_MODE (
    if exist "%VENV_DIR%" (
        echo [2/5] Reinstall mode: Deleting existing virtual environment...
        rmdir /s /q "%VENV_DIR%"
        if errorlevel 1 ( goto :error "Could not delete the existing '%VENV_DIR%' directory. Check for file locks." )
    )
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [2/5] Creating Python virtual environment in '%VENV_DIR%'...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if errorlevel 1 ( goto :error "Failed to create the virtual environment. Is Python installed and in your PATH?" )
    set "NEEDS_INSTALL=1"
) else (
    echo [2/5] Virtual environment already exists.
)

echo Activating virtual environment...
set "PATH=%CD%\%VENV_DIR%\Scripts;%PATH%"
echo [OK] Virtual environment is active.
echo.

:: --- [3/5] Installing/Verifying Dependencies ---
if defined NEEDS_INSTALL or defined REINSTALL_MODE (
    echo [3/5] Installing dependencies... This may take a few minutes.
    pip install --upgrade pip
    
    echo --- Installing project dependencies from pyproject.toml...
    pip install --upgrade .
    if errorlevel 1 ( goto :error "Failed to install project dependencies." )

) else (
    echo [3/5] Dependencies appear to be installed. Skipping installation.
)
echo [OK] Dependencies are ready.
echo.

:: --- [4/5] Running Diagnostics ---
echo [4/5] Running environment diagnostics...
python -m app.diagnostics
if errorlevel 1 (
    if defined DIAG_MODE (
        goto :error "One or more diagnostic checks failed. Please review the output above."
    ) else (
        echo.
        echo [WARNING] One or more diagnostic checks failed. The application might not run correctly.
        pause
    )
) else (
    echo [OK] All diagnostic checks passed.
)

if defined DIAG_MODE (
    echo.
    echo Diagnostic run complete.
    goto :end_success
)
echo.

:: --- [5/5] Launching Application ---
echo =======================================================
echo [5/5] Starting AssetPixelHand...
if defined PYTHON_ARGS (
    echo    (Debug Mode Enabled: Log level will be set to DEBUG)
)
echo =======================================================
echo.

if defined PROFILE_MODE (
    echo --- RUNNING WITH CPROFILE ---
    echo The application may run slower. After you finish the scan and close the app,
    echo a file 'app_data\scan_profile.pstats' will be created.
    echo.
    python -m cProfile -o "app_data\scan_profile.pstats" main.py %PYTHON_ARGS%
) else (
    python main.py %PYTHON_ARGS%
)

if errorlevel 1 (
    goto :error "The application exited unexpectedly. Check the 'app_data\app_log.txt' file for details."
)

echo.
echo =======================================================
if defined PROFILE_MODE (
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