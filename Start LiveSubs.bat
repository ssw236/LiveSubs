@echo off
cd /d "%~dp0"
py -V:3.12 livesubs.py --model large-v3 %*
pause
