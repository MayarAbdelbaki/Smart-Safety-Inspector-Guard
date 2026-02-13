@echo off
REM Connection Test Script for PC
REM Run this on your PC to test connectivity to autocar

set AUTOCAR_IP=192.168.101.101
set AUTOCAR_PORT=5001

echo ==========================================
echo Testing Connection to Autocar
echo ==========================================
echo Autocar IP: %AUTOCAR_IP%
echo Autocar Port: %AUTOCAR_PORT%
echo.

REM Test 1: Ping
echo Test 1: Ping to autocar...
ping -n 3 %AUTOCAR_IP% >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Ping successful
) else (
    echo [FAIL] Ping failed - Autocar is not reachable
    echo   Check: Are PC and autocar on the same network?
    pause
    exit /b 1
)
echo.

REM Test 2: HTTP endpoint
echo Test 2: Testing message receiver endpoint...
curl -s --max-time 5 http://%AUTOCAR_IP%:%AUTOCAR_PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Message receiver endpoint is responding
) else (
    echo [FAIL] Message receiver endpoint failed
    echo   Check: Is autocar running face_capture.py?
    pause
    exit /b 1
)
echo.

echo ==========================================
echo All tests passed! Connection is working.
echo ==========================================
pause
