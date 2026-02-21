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

:: Find first available port starting from 7860
setlocal enabledelayedexpansion
set APP_PORT=
for /L %%P in (7860,1,7960) do (
    if not defined APP_PORT (
        powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',%%P);$c.Close();exit 0}catch{exit 1}" >nul 2>&1
        if errorlevel 1 set APP_PORT=%%P
    )
)
if not defined APP_PORT set APP_PORT=7860
echo [OK] Using port !APP_PORT!
echo.

:: Open browser after 5 seconds on the detected port
start /b cmd /c "timeout /t 5 /nobreak > nul && start http://localhost:!APP_PORT!"

python app.py

pause
