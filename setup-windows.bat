@echo off
REM One-time setup for running wizAi against the live game on Windows.
REM
REM Creates .venv next to this file and installs everything the live path
REM needs. Safe to re-run -- pip will just report things already present.
REM
REM Why the fork: Wizard101 patched and the autobot function's prologue
REM changed, so the wizwalker vendored in Deimos/libs no longer finds its
REM signature and hook installation fails with PatternFailed.
REM LaurenzLikeThat's fork tracks the current build. It is a drop-in --
REM same package name, same Python floor, same pure-Python dependencies,
REM no Rust and no build tools.

setlocal
cd /d "%~dp0"

echo.
echo === wizAi live setup =========================================
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: git is not on PATH. Install it from https://git-scm.com
    echo        and reopen this window.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: could not create the venv. Is Python 3.11+ on PATH?
        echo        Check with:  python --version
        pause
        exit /b 1
    )
) else (
    echo .venv already exists, reusing it.
)

set PY=.venv\Scripts\python.exe

echo.
echo Installing wizwalker ^(LaurenzLikeThat fork^) ...
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip uninstall -y wizwalker >nul 2>&1
"%PY%" -m pip install "git+https://github.com/LaurenzLikeThat/wizwalker"
if errorlevel 1 (
    echo.
    echo ERROR: installing the wizwalker fork failed. See the output above.
    pause
    exit /b 1
)

echo.
echo Installing wizAi's own requirements ...
"%PY%" -m pip install --quiet numpy PyQt6
if errorlevel 1 (
    echo ERROR: installing numpy/PyQt6 failed.
    pause
    exit /b 1
)

echo.
echo Installing Deimos's questing requirements ...
REM Deimos/src/questing.py is the real navigator -- navmap teleports,
REM spiral doors, dungeon entry, NPC talking -- and composes with the
REM combat policy because auto_quest_solo no-ops during a duel. It needs
REM src.utils, which imports wizwalker.extensions.wizsprinter, hence
REM wizsprinter and its Python 3.13 floor. If this fails the bridge falls
REM back to its own lighter questing and everything else still works.
"%PY%" -m pip install --quiet "%~dp0Deimos\libs\wizsprinter" thefuzz loguru pyyaml requests pypresence pyperclip
if errorlevel 1 (
    echo.
    echo NOTE: Deimos's questing requirements did not install. That is not
    echo       fatal -- the bridge will use its own lighter questing, which
    echo       teleports and clicks dialogue but cannot navigate zones.
    echo       Everything else works. Python 3.13+ is required for this part.
    echo.
)

echo.
"%PY%" -c "import wizwalker, numpy, PyQt6; print('all imports OK')"
if errorlevel 1 (
    echo ERROR: something did not install cleanly.
    pause
    exit /b 1
)

echo.
echo === done =====================================================
echo.
echo   wizAi-gui.bat    the window ^(press Play live^)
echo   wizAi-live.bat   the console runner
echo.
echo Right-click either one, Create shortcut, and put the shortcut on
echo your desktop. The .bat itself has to stay in this folder -- it
echo finds .venv relative to its own location.
echo.
pause
