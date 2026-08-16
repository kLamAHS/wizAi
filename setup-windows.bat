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
REM shapely (with numpy) is the collision-teleport solver from the
REM Deimos 3.14 port: teleports are solved against the zone's collision
REM geometry for a landing spot that clears every wall. Without it
REM nothing breaks -- every teleport quietly falls back to the old
REM navmap behaviour -- which is exactly why it is installed here rather
REM than left optional: a silent fallback would make the fix invisible.
"%PY%" -m pip install --quiet numpy shapely PyQt6
if errorlevel 1 (
    echo ERROR: installing numpy/shapely/PyQt6 failed.
    pause
    exit /b 1
)

echo.
echo Installing Deimos's questing requirements ...
REM Deimos/src/questing.py is the real navigator -- navmap teleports,
REM spiral doors, dungeon entry, NPC talking -- and composes with the
REM combat policy because auto_quest_solo no-ops during a duel. The same
REM imports are what bot scripts (deimoslang) need.
REM
REM wizsprinter is NOT installed here. It is vendored at
REM Deimos\libs\wizsprinter and overlays into the wizwalker namespace, and
REM its pyproject declares requires-python >=3.13 -- so pip refuses it on
REM 3.11/3.12 and the whole line used to fail, taking the other packages
REM with it and leaving scripts broken with a misleading message. The
REM floor is declared metadata, not real: those sources compile on 3.11.
REM deimos_bridge/deimos_path.py puts them on the path at import time
REM instead, so only genuine PyPI packages are installed below. `lark` is
REM among them because wizsprinter needs it and nothing is resolving its
REM dependencies for us now.
REM katsuba + wiztype belong to the collision teleport's PRECISE layer
REM (exact teleporter-pad/boat colliders and player radius). That layer
REM only switches on when upstream Deimos's Rust extension `kinif` is
REM ALSO importable, which is not on PyPI -- so these two are installed
REM ready for it, and until then entity_collision quietly reports no
REM colliders and collision teleports work without that refinement.
"%PY%" -m pip install --quiet lark thefuzz loguru pyyaml requests pypresence pyperclip katsuba wiztype
if errorlevel 1 (
    echo.
    echo NOTE: Deimos's questing requirements did not install. That is not
    echo       fatal -- the bridge will use its own lighter questing, which
    echo       teleports and clicks dialogue but cannot navigate zones, and
    echo       bot scripts will be unavailable. Everything else works.
    echo.
)

echo.
"%PY%" -c "import wizwalker, numpy, shapely, PyQt6; print('all imports OK')"
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
