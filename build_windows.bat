@echo off
REM Build a single self-contained MidnightTrapsfRb.exe on Windows.
REM Requires the project dependencies plus a checkout of MOTorNOT as a sibling
REM directory (..\MOTorNOT) and pyinstaller (see requirements.txt).
REM The finished file is dist\MidnightTrapsfRb.exe -- copy it anywhere and run.

echo Building MidnightTrapsfRb.exe ...
pyinstaller --noconfirm --clean MidnightTrapsfRb.spec
if errorlevel 1 (
    echo.
    echo Build FAILED.
    exit /b 1
)
echo.
echo Done -- see dist\MidnightTrapsfRb.exe
