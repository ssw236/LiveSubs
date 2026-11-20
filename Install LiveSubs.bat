@echo off
setlocal
title LiveSubs setup
cd /d "%~dp0"
echo.
echo  ===== LiveSubs setup =====
echo.
echo  [1/4] Finding Python 3.12...
call :findpython
if not defined PY call :getpython
if not defined PY goto :nopython
echo        Using %PY%

rem First install only: pick the translator, the biggest download
set "TRARG="
if exist settings.json goto :picked
echo.
echo  Which translator do you want? You can switch later in the app.
echo    1 = Best quality         Qwen3 14B - 9 GB download, for graphics cards with 12 GB or more
echo    2 = Smaller and faster   Qwen3 8B  - 5 GB download, for graphics cards with 8 GB or more
choice /c 12 /n /t 60 /d 1 /m "  Press 1 or 2: "
if errorlevel 2 (set "TRARG=--translator qwen3-8b") else (set "TRARG=--translator qwen3-14b")
:picked
echo.
echo  [2/4] Installing packages (can take a few minutes)...
"%PY%" -m pip install --disable-pip-version-check -q --upgrade -r requirements.txt || goto :fail
call :whispergpu
echo.
echo  [3/4] Downloading and testing the AI engine and models...
echo        (first time: about 11 GB, or 7 GB with the smaller translator. If it gets
echo         interrupted, just run this setup again and it carries on where it stopped.)
"%PY%" livesubs.pyw --prepare %TRARG% || goto :fail
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
echo  Couldn't install Python 3.12 automatically.
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

:whispergpu
rem Optional: GPU libraries for the Whisper speech engines. The default setup doesn't need them: its speech
rem recognition already runs on the graphics card through the AI engine.
where nvidia-smi >nul 2>&1 || exit /b 0
"%PY%" -m pip show -q nvidia-cudnn-cu12 >nul 2>&1 && goto :whispergpu_install
echo.
echo  Optional: GPU speed-up for the Whisper speech engines - 1.7 GB more.
echo  Not needed for the default setup, which already uses your graphics card.
choice /c YN /n /t 20 /d N /m "  Add it? Press Y or N: "
if errorlevel 2 exit /b 0
:whispergpu_install
echo        Installing the GPU libraries for Whisper...
"%PY%" -m pip install --disable-pip-version-check -q nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 || exit /b 0
"%PY%" -m pip install --disable-pip-version-check -q "sherpa-onnx==2.0.1+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
if errorlevel 1 echo        (GPU version of the speech engine unavailable - SenseVoice will use the CPU)
exit /b 0

:findpython
rem Sets PY to a working Python 3.12, or leaves it empty.
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
if not defined PY exit /b 0
"%PY%" -c "import sys; assert sys.version_info[:2]==(3,12)" >nul 2>&1 || set "PY="
exit /b 0

:getpython
rem Downloads Python's official installer from python.org and installs it for this Windows user only,
rem so no admin rights are needed. 3.12.10 is the last Python 3.12 release with a Windows installer.
set "PYURL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
set "PYINST=%TEMP%\livesubs-python-3.12.10-amd64.exe"
echo        Python 3.12 isn't installed - downloading it from python.org (26 MB)...
del /q "%PYINST%" >nul 2>&1
"%SystemRoot%\System32\curl.exe" -L --fail --retry 3 --progress-bar -o "%PYINST%" "%PYURL%"
if errorlevel 1 (
  rem no curl, or it failed: try again with PowerShell
  del /q "%PYINST%" >nul 2>&1
  powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = 'Tls12'; $ProgressPreference = 'SilentlyContinue'; try { Invoke-WebRequest -UseBasicParsing -Uri $env:PYURL -OutFile $env:PYINST } catch { Remove-Item -ErrorAction SilentlyContinue -LiteralPath $env:PYINST }"
)
if not exist "%PYINST%" (
  echo        The download didn't work - check your internet connection.
  exit /b 0
)
rem Only run it if it really is Python's installer, signed by the Python Software Foundation
powershell -NoProfile -Command "$s = Get-AuthenticodeSignature -LiteralPath $env:PYINST; if ($s.Status -eq 'Valid' -and $s.SignerCertificate.Subject -like '*Python Software Foundation*') { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo        The downloaded installer couldn't be verified, so it wasn't run.
  del /q "%PYINST%" >nul 2>&1
  exit /b 0
)
echo        Installing Python 3.12 for your Windows user (no admin needed, about a minute)...
start "" /wait "%PYINST%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0 Include_test=0 Include_doc=0 Shortcuts=0
del /q "%PYINST%" >nul 2>&1
call :findpython
if defined PY echo        Python 3.12 installed.
exit /b 0
