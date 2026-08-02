@echo off
REM Run a live fight from the console, printing each decision.
REM
REM Pass any run_live argument through, e.g.
REM     wizAi-live.bat --school ice --fights 5
REM     wizAi-live.bat --school fire --policy blade-stack
REM     wizAi-live.bat --school fire --wizards 4
REM
REM --wizards drives up to four running clients as a hivemind: each round
REM every wizard waits for the others and then chooses against a board
REM that already carries what the rest of the party committed to. You
REM need one logged-in client per wizard.
REM
REM With no arguments it uses the school below -- edit it, or pass
REM --school on the command line.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv here. Run setup-windows.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    ".venv\Scripts\python.exe" -m deimos_bridge.run_live --school ice
) else (
    ".venv\Scripts\python.exe" -m deimos_bridge.run_live %*
)
pause
