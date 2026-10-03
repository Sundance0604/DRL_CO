@echo off
setlocal DisableDelayedExpansion
title DRL CO - Experiment Platform
set "DRL_LAUNCH_ROOT=%~dp0"
set "DRL_LAUNCH_BROWSER=--open-browser"
set "DRL_LAUNCH_NO_PAUSE="

:arguments
if "%~1"=="" goto change_directory
if /i "%~1"=="--no-browser" goto no_browser
if /i "%~1"=="--no-pause" goto no_pause
echo [ERROR] Supported options: --no-browser --no-pause
exit /b 2

:no_browser
set "DRL_LAUNCH_BROWSER="
shift
goto arguments

:no_pause
set "DRL_LAUNCH_NO_PAUSE=1"
shift
goto arguments

:change_directory
pushd "%DRL_LAUNCH_ROOT%"
if errorlevel 1 goto directory_failed

if not exist ".venv\Scripts\python.exe" goto missing_python
if not exist "frontend\dist\index.html" goto missing_ui

echo Starting DRL CO. An existing service will be reused.
".venv\Scripts\python.exe" -B -m experiment_core.launch start %DRL_LAUNCH_BROWSER%
set "DRL_LAUNCH_EXIT=%ERRORLEVEL%"
if not "%DRL_LAUNCH_EXIT%"=="0" goto launch_failed
echo.
echo The platform is ready. Its local URL is shown above.
echo You may close this window; the background service keeps running.
echo To stop it, run scripts\stop.ps1 from the project folder.
goto finish

:missing_python
echo [ERROR] Project Python environment is missing.
echo First run .\scripts\setup.ps1 in PowerShell from this project folder.
set "DRL_LAUNCH_EXIT=1"
goto finish

:missing_ui
echo [ERROR] Built web interface is missing: frontend\dist\index.html
echo First run .\scripts\setup.ps1, or run npm run build inside frontend.
set "DRL_LAUNCH_EXIT=1"
goto finish

:launch_failed
echo.
echo [ERROR] Startup did not complete. The error is shown above.
echo Check workspace\server.stderr.log for server errors.
echo If dependencies are missing, run .\scripts\setup.ps1 first.
goto finish

:directory_failed
echo [ERROR] Cannot open the project folder.
if not defined DRL_LAUNCH_NO_PAUSE pause
exit /b 1

:finish
if not defined DRL_LAUNCH_NO_PAUSE pause
popd
exit /b %DRL_LAUNCH_EXIT%
