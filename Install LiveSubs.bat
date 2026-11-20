@echo off
setlocal
title LiveSubs setup
cd /d "%~dp0"
echo.
echo  ===== LiveSubs setup =====
echo.
echo  [1/4] Finding Python 3.12...
set "PY="
rem Python install manager location, then the classic python.org installer locations
for %%P in (
  "%LOCALAPPDATA%\Python\pythoncore-3.12-64\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%ProgramFiles%\Python312\python.exe"
  "C:\Python312\python.exe"
) do if not defined PY if exist %%P set "PY=%%~P"
rem Fall back to the py launcher if it exists
if not defined PY (
  where py >nul 2>&1 && for /f "delims=" %%E in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%E"
)
if not defined PY goto :nopython
"%PY%" -c "import sys; assert sys.version_info[:2]==(3,12)" >nul 2>&1 || goto :nopython
echo        Using %PY%
echo.
echo  [2/4] Installing packages (can take a few minutes)...
"%PY%" -m pip install --disable-pip-version-check -q --upgrade -r requirements.txt || goto :fail
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :nogpu
echo        NVIDIA GPU found - installing GPU libraries...
"%PY%" -m pip install --disable-pip-version-check -q nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 || goto :fail
echo        Installing the GPU version of the speech engine...
"%PY%" -m pip install --disable-pip-version-check -q "sherpa-onnx==2.0.1+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
if errorlevel 1 echo        (GPU speech engine unavailable - the listener will use the CPU instead)
:nogpu
echo.
echo  [3/4] Downloading and testing the AI engine and models...
echo        (first time: about 19 GB - grab a drink. If it gets interrupted,
echo         just run this setup again and it carries on where it stopped.)
"%PY%" livesubs.pyw --prepare || goto :fail
echo.
echo  [4/4] Creating Desktop and Start menu shortcuts...
"%PY%" livesubs.pyw --install-shortcuts || goto :fail
echo.

echo  ===== Done! =====
echo  Open LiveSubs from the shortcut on your desktop.
echo.
pause
exit /b 0

:nopython
echo.
echo  Couldn't find Python 3.12 on this PC.
echo  Install it from the page that just opened: scroll to "Files", download the
echo  "Windows installer (64-bit)", run it, and TICK "Add python.exe to PATH".
echo  Then run this setup again.
start "" https://www.python.org/downloads/release/python-31210/
goto :fail

:fail
echo.
echo  Something went wrong - see the messages above for details.
echo.
pause
exit /b 1
