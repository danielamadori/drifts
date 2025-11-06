@echo off
REM Live monitoring of test progress
REM Run this in a SEPARATE terminal window while tests are running

cd /d C:\Users\danie\Projects\drifts

echo.
echo ========================================
echo  DATASET TEST MONITOR
echo ========================================
echo.
echo This will show live progress of tests.
echo Run this in a separate window while
echo tests are running in another terminal.
echo.
echo Press any key to start monitoring...
pause > nul

python monitor_live.py

pause

