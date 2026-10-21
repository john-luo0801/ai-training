@echo off
setlocal
cd /d "%~dp0"
title Neural CSV

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py" was not found.
    echo Install Python 3 from https://www.python.org and run this file again.
    pause
    exit /b 1
)

if exist "vendor\python\*.whl" (
    echo Checking dependencies from vendor\python ...
    py -3 -m pip install --quiet --no-index --find-links vendor\python -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Could not install packages from vendor\python.
        echo Try install_deps.bat, then run this file again.
        pause
        exit /b 1
    )
) else (
    echo [WARN] vendor\python has no wheels. Falling back to the internet...
    py -3 -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency install failed.
        pause
        exit /b 1
    )
)

echo Starting Neural CSV...
py -3 main.py
endlocal
