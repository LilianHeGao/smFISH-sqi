@echo off
REM ============================================================
REM  Batch SQI pipeline: auto-discover FOVs, randomly select N,
REM  run pipeline on each, generate tissue overview.
REM
REM  Edit DATA_FLD, CACHE_ROOT, OUT_ROOT, FOV_NUM below.
REM ============================================================

set DATA_FLD=\\192.168.0.116\durian3\Lilian\022425_FTD_smFISH_MBP_NRGN\coverslip2_FTD_group1\MBP_NRGN_set2
set CACHE_ROOT=\\192.168.0.116\durian3\Lilian\merfish_sqi_cache
set OUT_ROOT=output\022425_FTD_smFISH_MBP_NRGN\coverslip2_FTD_group1\MBP_NRGN_set2
set FOV_NUM=10

python scripts\run_batch_fovs.py ^
    --data_fld   "%DATA_FLD%" ^
    --cache_root "%CACHE_ROOT%" ^
    --out_root   "%OUT_ROOT%" ^
    --n_fovs %FOV_NUM% ^
    --seed 42

pause
