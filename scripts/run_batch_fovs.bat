@echo off
REM Batch run SQI pipeline for multiple FOVs.
REM Usage: run_batch_fovs.bat

set DATA_FLD=M:\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1
set CACHE_ROOT=\\192.168.0.73\Papaya13\Lilian\merfish_sqi_cache
set OUT_ROOT=output
set ZARR_BASE=\\192.168.0.73\Papaya13\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1

for %%F in (032 043 086 115) do (
    echo ============================================================
    echo Processing FOV %%F ...
    echo ============================================================
    python scripts\run_sqi_from_fov_zarr.py ^
        --fov_zarr   "%ZARR_BASE%\Conv_zscan1_%%F.zarr" ^
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
