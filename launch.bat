@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

:: --- Auto-detect Python: prefer venv, then PATH ---
if defined VIRTUAL_ENV (
    set "PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
) else if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found. Activate a venv or add python to PATH.
        exit /b 1
    )
    for /f "delims=" %%P in ('where python') do set "PYTHON=%%P"
)
set "PIO_CORE=C:\PIO"
set "FW_DIR=%ROOT%firmware\esp32s3_gimbal"
set "ENV_OK="
title TAFFF Gimbal Tracker
color 0F

call :env_check
if errorlevel 1 goto :fatal

:: ================================================================
::  Parse config.yaml defaults (Phase 1)
:: ================================================================
call :parse_config
if errorlevel 1 goto :fatal

:: ================================================================
::  MAIN MENU
:: ================================================================
:menu
cls
call :parse_config
if errorlevel 1 goto :fatal
echo.
echo  ============================================================
echo    TAFFF Vision Gimbal Tracker
echo    GPU: GTX 1650 Super  ^| Precision: FP16
echo  ============================================================
echo.
:: ── Status dashboard (Phase 2) ──
echo    Mount: on_gimbal ^| Comm: !CFG_CHANNEL! (!CFG_COMM_DETAIL!) ^| Source: !CFG_SOURCE! ^| FOV: !CFG_FOV!
set "ENG_H=[--]" & set "ENG_D=[--]"
if exist "%ROOT%engines\yolo26n-person-17pose.engine" set "ENG_H=[OK]"
if exist "%ROOT%engines\enhanced-dog-24pose.engine"   set "ENG_D=[OK]"
echo    Engines: !ENG_H! Human   !ENG_D! Dog
echo.
echo    1 - Track Human        [quick launch]
echo    2 - Track Dog          [quick launch]
echo    3 - Track Custom       [interactive setup]
echo    4 - Calibrate          [gimbal calibration]
echo    5 - Export Engines     [TensorRT build]
echo    6 - Diagnostics       [laser debug, camera test]
echo    7 - Firmware           [build / upload / monitor]
echo    8 - Edit Config        [open config.yaml]
echo    0 - Exit
echo.
set "MC="
set /p "MC=  Select: "
if "%MC%"=="1" goto :quick_human
if "%MC%"=="2" goto :quick_dog
if "%MC%"=="3" goto :custom_track
if "%MC%"=="4" goto :calibrate_menu
if "%MC%"=="5" goto :export_menu
if "%MC%"=="6" goto :diagnostics_menu
if "%MC%"=="7" goto :firmware_menu
if "%MC%"=="8" goto :edit_config
if "%MC%"=="0" goto :exit_clean
goto :menu

:: ================================================================
::  1/2 - QUICK TRACK
:: ================================================================
:quick_human
set "QT_TARGET=human"
set "QT_ENGINE=%ROOT%engines\yolo26n-person-17pose.engine"
goto :quick_track

:quick_dog
set "QT_TARGET=dog"
set "QT_ENGINE=%ROOT%engines\enhanced-dog-24pose.engine"
goto :quick_track

:quick_track
cls
echo.
echo  Quick Track: %QT_TARGET%
echo  ----------------------------------------
echo.
if not exist "%QT_ENGINE%" (
    echo  Engine missing: %QT_ENGINE%
    echo  Exporting now ... this takes 2-5 minutes.
    echo.
    "%PYTHON%" "%ROOT%scripts\export_engines.py" --target %QT_TARGET%
    if errorlevel 1 (
        echo.
        echo  [FAIL] Engine export failed.
        pause
        goto :menu
    )
)
echo  [OK] Engine ready
echo  Mount:   on_gimbal
echo  Comm:    !CFG_CHANNEL! (!CFG_COMM_DETAIL!)
echo  Source:  !CFG_SOURCE!
echo  FOV:     !CFG_FOV!
echo.
echo  Launching %QT_TARGET% tracker...
echo  Hotkeys: ESC = quit, M = auto/manual, P = laser toggle, O = relay pulse, H = help overlay
echo.
:qt_launch
"%PYTHON%" -m src.main --mode camera --target %QT_TARGET% --source "!CFG_SOURCE!" --config config.yaml
echo.
echo  Tracker exited with code %ERRORLEVEL%.
echo.
echo    R - Relaunch   M - Menu
set "QT_AGAIN="
set /p "QT_AGAIN=  > "
if /I "!QT_AGAIN!"=="R" goto :qt_launch
goto :menu

:: ================================================================
::  3 - CUSTOM TRACK
:: ================================================================
:custom_track
cls
echo.
echo  Custom Track Setup
echo  ----------------------------------------
echo.
:: Phase 6: Offer to reuse last settings
if exist "logs\.last_launch.cfg" (
    echo  Last session found. Press L to reuse, or any other key for new setup.
    set "CT_REUSE="
    set /p "CT_REUSE=  > "
    if /I "!CT_REUSE!"=="L" goto :ct_reuse_last
)
goto :ct_fresh

:ct_reuse_last
:: Load saved settings
for /f "usebackq tokens=1,* delims==" %%A in ("logs\.last_launch.cfg") do (
    set "%%A=%%B"
)
set "MODE_ARG=!MODE!"
set "TARGET_ARG=!TARGET!"
set "SOURCE_ARG=!SOURCE!"
call :resolve_engine_path "%TARGET_ARG%"
call :prompt_engine_ready "%TARGET_ARG%" "!CE_PATH!"
if errorlevel 1 (
    pause
    goto :menu
)
goto :ct_summary

:ct_fresh
call :select_mode
call :select_target
call :select_source
set "COMM_ARGS="
set "COMM_TYPE="
set "FOV_ARG="
if /I not "%MODE_ARG%"=="camera" goto :ct_skip_comm
call :select_comm
call :select_fov
:ct_skip_comm
call :select_options
call :resolve_engine_path "%TARGET_ARG%"
call :prompt_engine_ready "%TARGET_ARG%" "!CE_PATH!"
if errorlevel 1 (
    pause
    goto :menu
)
:ct_summary
echo.
echo  ========================================
echo    Mode:    %MODE_ARG%
echo    Target:  %TARGET_ARG%
echo    Mount:   on_gimbal
echo    Source:  %SOURCE_ARG%
if defined COMM_TYPE echo    Comm:    !COMM_TYPE!
if defined FOV_ARG echo    FOV:     !FOV_ARG! deg
if defined EXTRA_ARGS echo    Flags:   %EXTRA_ARGS%
echo    Hotkeys: ESC = quit, M = auto/manual, P = laser toggle, O = relay pulse, H = help
echo  ========================================
echo.
set "CT_CMD="%PYTHON%" -m src.main --mode %MODE_ARG% --target %TARGET_ARG% --source "%SOURCE_ARG%" --config config.yaml %COMM_ARGS% %FOV_ARGS% %EXTRA_ARGS%"
:ct_launch
"%PYTHON%" -m src.main --mode %MODE_ARG% --target %TARGET_ARG% --source "%SOURCE_ARG%" --config config.yaml %COMM_ARGS% %FOV_ARGS% %EXTRA_ARGS%
echo.
echo  Tracker exited with code %ERRORLEVEL%.
:: Save last-used settings (Phase 6)
(
    echo MODE=%MODE_ARG%
    echo TARGET=%TARGET_ARG%
    echo SOURCE=%SOURCE_ARG%
    echo COMM_ARGS=%COMM_ARGS%
    echo COMM_TYPE=!COMM_TYPE!
    echo FOV_ARG=!FOV_ARG!
    echo FOV_ARGS=!FOV_ARGS!
    echo EXTRA_ARGS=%EXTRA_ARGS%
) > "logs\.last_launch.cfg" 2>nul
echo.
echo    R - Relaunch   M - Menu
set "CT_AGAIN="
set /p "CT_AGAIN=  > "
if /I "!CT_AGAIN!"=="R" goto :ct_launch
goto :menu

:: ================================================================
::  4 - CALIBRATE
:: ================================================================
:calibrate_menu
cls
echo.
echo  Gimbal Calibration
echo  ----------------------------------------
echo  Comm: !CFG_CHANNEL! (!CFG_COMM_DETAIL!) ^| Source: !CFG_SOURCE!
echo.
echo    1 - Calibrate            [interactive jog + auto-converge]
echo    2 - Auto Calibrate       [skip manual jog, auto-converge only]
echo    3 - Reset Offsets        [zero all calibration data]
echo    0 - Back
echo.
set "CC="
set /p "CC=  Select: "
if "%CC%"=="0" goto :menu
if "%CC%"=="1" set "CAL_FLAGS=" & goto :cal_setup
if "%CC%"=="2" set "CAL_FLAGS=--auto" & goto :cal_setup
if "%CC%"=="3" set "CAL_FLAGS=--reset" & goto :cal_setup
goto :calibrate_menu

:cal_setup
if "!CFG_SERIAL_PORT!"=="" (
    echo.
    echo  [FAIL] Calibration requires a serial port in config.yaml.
    pause
    goto :menu
)
:: Calibration packets are USB-only in firmware.
    set "CAL_COMM=--port !CFG_SERIAL_PORT! --baud !CFG_BAUD_RATE!"

set "CAL_SRC="
echo.
    echo    Calibration uses serial: !CFG_SERIAL_PORT! @ !CFG_BAUD_RATE!
echo    Camera source (default: !CFG_SOURCE!):
set /p "CAL_SRC=    > "
if not defined CAL_SRC set "CAL_SRC=!CFG_SOURCE!"
echo.
echo  Running calibration... %CAL_FLAGS%
echo.
    "%PYTHON%" "%ROOT%scripts\calibrate.py" %CAL_COMM% --source "%CAL_SRC%" --backend !CFG_BACKEND! %CAL_FLAGS%
echo.
echo  Calibration finished with code %ERRORLEVEL%.
pause
goto :menu

:: ================================================================
::  5 - EXPORT ENGINES
:: ================================================================
:export_menu
cls
echo.
echo  TensorRT Engine Export
echo  ----------------------------------------
echo.
echo    1 - Human  (yolo26n-person-17pose)
echo    2 - Dog    (enhanced-dog-24pose)
echo    3 - All
echo    0 - Back
echo.
set "EC="
set /p "EC=  Select: "
if "%EC%"=="0" goto :menu
if "%EC%"=="1" set "EXP_TGT=human" & goto :do_export
if "%EC%"=="2" set "EXP_TGT=dog"   & goto :do_export
if "%EC%"=="3" set "EXP_TGT=all"   & goto :do_export
goto :export_menu

:do_export
echo.
echo  Exporting %EXP_TGT% (FP16, 640x640)...
echo  This takes 2-5 min per model. Do not close.
echo.
"%PYTHON%" "%ROOT%scripts\export_engines.py" --target %EXP_TGT%
echo.
if errorlevel 1 (
    echo  [FAIL] Export failed.
) else (
    echo  [OK] Export complete.
)
pause
goto :menu

:: ================================================================
::  6 - DIAGNOSTICS
:: ================================================================
:diagnostics_menu
cls
echo.
echo  Diagnostics
echo  ----------------------------------------
echo.
echo    1 - Laser Debug          [live 4-panel HSV view]
echo    2 - Laser Capture        [single frame analysis]
echo    3 - Camera Preview       [test camera feed]
echo    4 - Run Tests            [pytest test suite]
echo    0 - Back
echo.
set "DC="
set /p "DC=  Select: "
if "%DC%"=="0" goto :menu
if "%DC%"=="1" goto :diag_laser_debug
if "%DC%"=="2" goto :diag_laser_capture
if "%DC%"=="3" goto :diag_camera
if "%DC%"=="4" goto :diag_tests
goto :diagnostics_menu

:diag_laser_debug
echo.
echo  Laser debug (press Q to quit)...
echo.
"%PYTHON%" "%ROOT%scripts\laser_debug.py"
pause
goto :menu

:diag_laser_capture
echo.
echo  Capturing laser frame...
echo.
"%PYTHON%" "%ROOT%scripts\laser_capture.py"
pause
goto :menu

:diag_camera
set "CAM_IDX="
echo.
echo  Camera source (default: !CFG_SOURCE!):
set /p "CAM_IDX=  > "
if not defined CAM_IDX set "CAM_IDX=!CFG_SOURCE!"
echo.
echo  Opening source %CAM_IDX% with backend !CFG_BACKEND! (press Q or ESC to close)...
echo.
"%PYTHON%" -c "import cv2,sys;n='Preview';src=sys.argv[1];src=int(src) if src.isdigit() else src;backends={'auto':cv2.CAP_ANY,'dshow':cv2.CAP_DSHOW,'msmf':cv2.CAP_MSMF,'ffmpeg':cv2.CAP_FFMPEG};c=cv2.VideoCapture(src,backends.get(sys.argv[2],cv2.CAP_ANY));exec('while c.isOpened():\n s,f=c.read()\n if not s:break\n cv2.imshow(n,f)\n if cv2.waitKey(1)^&0xFF in(27,113):break');c.release();cv2.destroyAllWindows()" "%CAM_IDX%" !CFG_BACKEND!
pause
goto :menu

:diag_tests
echo.
echo  Running test suite...
echo.
"%PYTHON%" -m pytest tests/ -q --tb=short
echo.
pause
goto :menu

:: ================================================================
::  7 - FIRMWARE
:: ================================================================
:firmware_menu
cls
echo.
echo  ESP32-S3 Firmware
echo  ----------------------------------------
echo.
echo    1 - Build
echo    2 - Build + Upload
echo    3 - Serial Monitor
echo    4 - Clean + Rebuild
echo    0 - Back
echo.
set "FC="
set /p "FC=  Select: "
if "%FC%"=="0" goto :menu
if "%FC%"=="1" goto :fw_build
if "%FC%"=="2" goto :fw_upload
if "%FC%"=="3" goto :fw_monitor
if "%FC%"=="4" goto :fw_rebuild
goto :firmware_menu

:fw_build
echo.
echo  Building firmware...
echo.
pushd "%FW_DIR%"
set "PLATFORMIO_CORE_DIR=%PIO_CORE%"
pio run -e esp32s3
popd
echo.
pause
goto :menu

:fw_upload
echo.
echo  Building and uploading...
echo.
pushd "%FW_DIR%"
set "PLATFORMIO_CORE_DIR=%PIO_CORE%"
pio run -e esp32s3 -t upload
popd
echo.
pause
goto :menu

:fw_monitor
echo.
echo  Serial monitor (Ctrl+C to exit)...
echo.
pushd "%FW_DIR%"
set "PLATFORMIO_CORE_DIR=%PIO_CORE%"
pio run -e esp32s3 -t monitor
popd
echo.
pause
goto :menu

:fw_rebuild
echo.
echo  Clean + rebuild...
echo.
pushd "%FW_DIR%"
set "PLATFORMIO_CORE_DIR=%PIO_CORE%"
pio run -e esp32s3 -t clean
pio run -e esp32s3
popd
echo.
pause
goto :menu

:: ================================================================
::  8 - EDIT CONFIG
:: ================================================================
:edit_config
where code >nul 2>&1
if not errorlevel 1 (
    start "" code "%ROOT%config.yaml"
) else (
    start "" notepad "%ROOT%config.yaml"
)
goto :menu

:: ================================================================
::  EXIT / FATAL
:: ================================================================
:exit_clean
echo.
echo  Goodbye.
endlocal & exit /b 0

:fatal
echo.
echo  ============================================================
echo  Launch aborted. Fix the issue above and try again.
echo  ============================================================
pause
endlocal & exit /b 1

:: ================================================================
::  SUBROUTINES
:: ================================================================

:: -- Environment Check (cached) --
:env_check
if defined ENV_OK exit /b 0
cls
echo.
echo  Checking environment...
echo.
if not exist "%PYTHON%" (
    echo  [FAIL] Python venv not found: %PYTHON%
    exit /b 1
)
echo  [OK] Python venv
"%PYTHON%" -c "import torch;assert torch.cuda.is_available();print('  [OK] CUDA:',torch.cuda.get_device_name(0))" 2>nul
if errorlevel 1 (
    echo  [FAIL] CUDA not available.
    exit /b 1
)
"%PYTHON%" -c "import tensorrt;print('  [OK] TensorRT:',tensorrt.__version__)" 2>nul
if errorlevel 1 (
    echo  [FAIL] TensorRT missing. pip install tensorrt
    exit /b 1
)
"%PYTHON%" -c "import cv2;print('  [OK] OpenCV:',cv2.__version__)" 2>nul
if errorlevel 1 (
    echo  [FAIL] OpenCV missing. pip install opencv-python
    exit /b 1
)
if not exist "config.yaml" (
    echo  [FAIL] config.yaml not found.
    exit /b 1
)
echo  [OK] config.yaml
echo.
echo  Environment OK.
timeout /t 1 >nul
set "ENV_OK=1"
exit /b 0

:: -- Engine Check helpers --
:resolve_engine_path
if /I "%~1"=="human" (
    set "CE_PATH=%ROOT%engines\yolo26n-person-17pose.engine"
) else (
    set "CE_PATH=%ROOT%engines\enhanced-dog-24pose.engine"
)
exit /b 0

:prompt_engine_ready
if exist "%~2" (
    echo  [OK] Engine: %~2
    exit /b 0
)
echo  Engine missing: %~2
set "CE_YN="
set /p "CE_YN=  Export now? [Y/n]: "
if /I "%CE_YN%"=="n" exit /b 1
echo.
"%PYTHON%" "%ROOT%scripts\export_engines.py" --target %~1
if errorlevel 1 (
    echo  [FAIL] Export failed.
    exit /b 1
)
echo  [OK] Engine: %~2
exit /b 0

:: -- Select Mode --
:select_mode
echo  Mode:
echo    1 - Camera   2 - Video
echo.
:sm_loop
set "MODE_CHOICE="
set /p "MODE_CHOICE=  > "
if "%MODE_CHOICE%"=="1" ( set "MODE_ARG=camera" & exit /b 0 )
if "%MODE_CHOICE%"=="2" ( set "MODE_ARG=video"  & exit /b 0 )
goto :sm_loop

:: -- Select Target --
:select_target
echo.
echo  Target:
echo    1 - Human (17 keypoints)   2 - Dog (24 keypoints)
echo.
:st_loop
set "TARGET_CHOICE="
set /p "TARGET_CHOICE=  > "
if "%TARGET_CHOICE%"=="1" ( set "TARGET_ARG=human" & exit /b 0 )
if "%TARGET_CHOICE%"=="2" ( set "TARGET_ARG=dog"   & exit /b 0 )
goto :st_loop

:: -- Select Source --
:select_source
echo.
if /I "%MODE_ARG%"=="video" goto :ss_video
echo  Camera index (default: !CFG_SOURCE!):
set "SOURCE_ARG="
set /p "SOURCE_ARG=  > "
if not defined SOURCE_ARG set "SOURCE_ARG=!CFG_SOURCE!"
exit /b 0

:ss_video
echo  Available videos in videos\:
set "VID_N=0"
for %%F in (videos\*.mp4 videos\*.avi videos\*.mkv videos\*.mov) do (
    set /a "VID_N+=1"
    set "VID_!VID_N!=%%F"
    echo    !VID_N! - %%~nxF
)
if !VID_N!==0 echo    (none found)
echo.
echo  Enter a number, filename, or full path/URL:
:ss_loop
set "SOURCE_ARG="
set /p "SOURCE_ARG=  > "
if not defined SOURCE_ARG goto :ss_loop
:: Check if user typed a number matching a listed video
set "VID_CHECK=!VID_%SOURCE_ARG%!"
if defined VID_CHECK (
    set "SOURCE_ARG=%ROOT%!VID_CHECK!"
    exit /b 0
)
if exist "%SOURCE_ARG%" exit /b 0
if exist "videos\%SOURCE_ARG%" (
    set "SOURCE_ARG=%ROOT%videos\%SOURCE_ARG%"
    exit /b 0
)
:: Allow URLs through without file existence check
echo "%SOURCE_ARG%" | findstr /i "http:// https:// rtsp://" >nul && exit /b 0
echo  Not found: %SOURCE_ARG%
goto :ss_loop

:: -- Select Comm --
:select_comm
echo.
echo  Gimbal output (config.yaml: !CFG_CHANNEL!):
if /I "!CFG_CHANNEL!"=="serial" (
    echo    1 - Serial [!CFG_SERIAL_PORT!]   2 - UDP   3 - Auto   4 - None
) else if /I "!CFG_CHANNEL!"=="udp" (
    echo    1 - Serial   2 - UDP [!CFG_UDP_HOST!:!CFG_UDP_PORT!]   3 - Auto   4 - None
) else (
    echo    1 - Serial   2 - UDP   3 - Auto [serial-^>udp]   4 - None
)
echo    Enter = use config.yaml default
echo.
:sc_loop
set "COMM_CHOICE="
set /p "COMM_CHOICE=  > "
if not defined COMM_CHOICE goto :sc_default
if "%COMM_CHOICE%"=="1" goto :sc_serial
if "%COMM_CHOICE%"=="2" goto :sc_udp
if "%COMM_CHOICE%"=="3" (
    set "COMM_ARGS="
    set "COMM_TYPE=auto (serial->udp)"
    exit /b 0
)
if "%COMM_CHOICE%"=="4" (
    set "COMM_ARGS=--no-comm"
    set "COMM_TYPE=none"
    exit /b 0
)
goto :sc_loop

:sc_default
:: Preserve config.yaml exactly with no CLI override.
set "COMM_ARGS="
if /I "!CFG_CHANNEL!"=="serial" (
    set "COMM_TYPE=config default: serial (!CFG_SERIAL_PORT!)"
) else if /I "!CFG_CHANNEL!"=="udp" (
    set "COMM_TYPE=config default: udp (!CFG_UDP_HOST!:!CFG_UDP_PORT!)"
) else (
    set "COMM_TYPE=config default: auto (serial->udp)"
)
exit /b 0

:sc_serial
set "SP="
echo    COM port (default: !CFG_SERIAL_PORT!):
set /p "SP=    > "
if not defined SP set "SP=!CFG_SERIAL_PORT!"
set "COMM_ARGS=--comm serial --serial-port %SP%"
set "COMM_TYPE=serial (%SP%)"
exit /b 0

:sc_udp
set "UH="
set "UP="
echo    Host (default: !CFG_UDP_HOST!):
set /p "UH=    > "
if not defined UH set "UH=!CFG_UDP_HOST!"
echo    Port (default: !CFG_UDP_PORT!):
set /p "UP=    > "
if not defined UP set "UP=!CFG_UDP_PORT!"
set "COMM_ARGS=--comm udp --udp-host %UH% --udp-port %UP%"
set "COMM_TYPE=udp (%UH%:%UP%)"
exit /b 0

:: -- Select FOV --
:select_fov
set "FOV_ARG="
set "FOV_ARGS="
echo.
echo  Camera FOV in degrees (Enter = keep !CFG_FOV!):
set "FOV_IN="
set /p "FOV_IN=  > "
if defined FOV_IN (
    set "FOV_ARG=!FOV_IN!"
    set "FOV_ARGS=--fov !FOV_IN!"
)
exit /b 0

:: -- Select Options --
:select_options
set "EXTRA_ARGS="
echo.
echo  Extra flags (Enter for none):
echo    --debug  --headless  --no-quit  --profile  --log-level DEBUG
set "OPT="
set /p "OPT=  > "
if defined OPT set "EXTRA_ARGS=%OPT%"
exit /b 0

:: -- Parse config.yaml into CFG_ variables --
:parse_config
set "CFG_CHANNEL=serial"
set "CFG_SERIAL_PORT=COM4"
set "CFG_BAUD_RATE=921600"
set "CFG_BACKEND=auto"
set "CFG_UDP_HOST=192.168.4.1"
set "CFG_UDP_PORT=6000"
set "CFG_FOV=?"
set "CFG_SOURCE=0"
set "CFG_TARGET=human"
set "CFG_COMM_DETAIL=COM4"
set "CFG_PARSE_OK="
:: Use the Python config loader so launcher defaults stay aligned with runtime behavior.
for /f "tokens=1,* delims==" %%A in ('"%PYTHON%" -m src.config_loader --launcher-env config.yaml 2^>nul') do (
    set "%%A=%%B"
    set "CFG_PARSE_OK=1"
)
if not defined CFG_PARSE_OK (
    echo.
    echo  [FAIL] Could not read config defaults from config.yaml.
    exit /b 1
)
exit /b 0