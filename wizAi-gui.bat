@echo off
REM Open the wizAi live combat lab.
REM
REM `%~dp0` is this file's own folder, so the shortcut you put on the
REM desktop keeps working and no path is hard-coded. The .bat itself has
REM to stay in the repository root.
REM
REM Uses python.exe rather than pythonw.exe deliberately: the console
REM stays open, so an import error that happens before the window exists
REM is visible instead of the app silently never appearing.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv here. Run setup-windows.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m deimos_bridge.gui %*
if errorlevel 1 pause
