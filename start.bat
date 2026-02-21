@echo off
cd /d "%~dp0"

echo ==========================================
echo   voice-echo Starting...
echo ==========================================
echo.

:: Find conda activate script
set CONDA_BAT=
if exist "D:\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=D:\miniconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
)

if "%CONDA_BAT%"=="" (
    echo [WARNING] conda activate script not found. Trying PATH...
    call conda activate main
) else (
    call "%CONDA_BAT%" main
)

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate conda env "main"
    echo Please check: conda env list
    echo.
    pause
    exit /b 1
)

echo [OK] conda env "main" activated
echo.

:: Open browser after 5 seconds
start /b cmd /c "timeout /t 5 /nobreak > nul && start http://localhost:7860"

python app.py

pause
