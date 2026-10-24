@echo off
rem ISL keyframe annotator. First run: creates a Python environment, installs opencv + pillow,
rem and runs the one-time setup (SSH key for the shared machine, shared lists, NAS check).
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%~dp0servers.json" goto missing
if not exist "%~dp0annotation_queue.csv" goto missing

if not exist "%PY%" (
    echo First run: creating the Python environment...
    python -m venv "%~dp0.venv" || goto nopython
    "%PY%" -m pip install -q -r requirements.txt || goto failed
)

if not exist "%~dp0.setup_done" (
    echo.
    echo One-time setup: connects this computer to the shared annotation machine and the NAS.
    "%PY%" server_videos.py setup || goto failed
    echo done> "%~dp0.setup_done"
)

"%PY%" app.py
goto :eof

:missing
echo servers.json and annotation_queue.csv must be in this folder:
echo   %~dp0
echo They are not in git. Get them from the team (see README.md, "Getting started").
pause
goto :eof

:nopython
echo Python 3 was not found. Install it from python.org (tick "Add python.exe to PATH"), then run start.bat again.
pause
goto :eof

:failed
echo.
echo Something failed, see the messages above. Fix it and run start.bat again.
pause
