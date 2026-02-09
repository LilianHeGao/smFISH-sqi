@echo off
REM Batch run SQI pipeline for multiple FOVs.
REM Usage: run_batch_fovs.bat

set DATA_FLD=\\192.168.0.73\Papaya13\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set11
set CACHE_ROOT=\\192.168.0.73\Papaya13\Lilian\merfish_sqi_cache
set OUT_ROOT=output\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set11
set ZSCAN_NUM=10
set FOV_LIST=057 101

for %%F in (%FOV_LIST%) do (
    echo ============================================================
    echo Processing FOV %%F ...
    echo ============================================================
    python scripts\run_sqi_from_fov_zarr.py ^
        --fov_zarr   "%DATA_FLD%\Conv_zscan%ZSCAN_NUM%_%%F.zarr" ^
        --data_fld   "%DATA_FLD%" ^
        --cache_root "%CACHE_ROOT%" ^
        --out_root   "%OUT_ROOT%"
    if errorlevel 1 (
        echo [ERROR] FOV %%F failed!
    ) else (
        echo [OK] FOV %%F done.
    )
    echo.
)

echo ============================================================
echo All FOVs complete.
echo ============================================================
pause
