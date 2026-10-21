@echo off
setlocal
cd /d "%~dp0"
title Install Neural CSV dependencies

echo ============================================================
echo  Installing packages from the local vendor\python registry
echo  (no internet connection required)
echo ============================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py" was not found.
    echo Install Python 3 from https://www.python.org and run this file again.
    goto :failed
)

if not exist "vendor\python\*.whl" (
    echo [ERROR] vendor\python has no wheel files.
    echo Run:  py -3 data\download_vendor_packages.py
    echo (that step needs the internet once), then try again.
    goto :failed
)

echo Installing from vendor\python ...
py -3 -m pip install --no-index --find-links vendor\python -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Done. You can now run RUN.bat or:  py -3 main.py
goto :end

:failed
echo.
echo [FAILED] Dependencies were not installed. See the message above.

:end
echo.
pause
endlocal
