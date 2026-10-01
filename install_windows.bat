@echo off
REM Windows installer (CPU simulator). Needs Python 3.11 or 3.12 from python.org ("Add python.exe to PATH" ticked).
REM For the RTX GPU use WSL2 and install_linux.sh instead (see SETUP.md).
cd /d "%~dp0"
py -3.12 --version >nul 2>&1 && (set PY=py -3.12) || (set PY=py -3.11)
%PY% -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r Code\requirements.txt || goto :error
python Code\run_all.py --check
echo.
echo Installed. Next time open a terminal here and run:
echo     .venv\Scripts\activate
echo     python Code\run_all.py               (simulation + figures)
echo     python Code\run_all.py --hardware    (also the real IBM quantum computer)
pause
exit /b 0
:error
echo Installation failed - see the message above and SETUP.md
pause
exit /b 1
