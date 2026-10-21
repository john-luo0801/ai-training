@echo off
setlocal
cd /d "%~dp0"
title Build NeuralCSV.exe

echo ============================================================
echo  Building NeuralCSV.exe from the current code
echo  Output: dist\NeuralCSV.exe
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
    echo then try again.
    goto :failed
)

echo [1/3] Installing dependencies from the local vendor\python registry...
py -3 -m pip install --no-index --find-links vendor\python -r requirements.txt pyinstaller
if errorlevel 1 goto :failed

echo.
echo [2/3] Building the executable with PyInstaller...
if exist build rmdir /s /q build
py -3 -m PyInstaller --noconfirm --clean --onefile --console --name NeuralCSV ^
    --add-data "neural_csv\templates;neural_csv\templates" ^
    --add-data "neural_csv\static;neural_csv\static" ^
    --collect-all sklearn ^
    main.py
if errorlevel 1 goto :failed

echo.
if not exist "dist\NeuralCSV.exe" (
    echo [ERROR] PyInstaller finished but dist\NeuralCSV.exe was not created.
    goto :failed
)

echo [3/3] Done.
echo.
echo Executable: %CD%\dist\NeuralCSV.exe
echo Double-click it to start Neural CSV. Your saved model and retained
echo training rows will be kept in a "models" folder next to the .exe,
echo so do not delete that folder unless you want to start over.
echo You can copy NeuralCSV.exe to another Windows PC - Python is not needed there.
goto :end

:failed
echo.
echo [FAILED] The build did not complete. Read the error above, fix it,
echo then run build_exe.bat again.

:end
echo.
pause
endlocal
