@echo off
REM ============================================================
REM  Compare SQI sanity-check AUCs across 3 conditions
REM  (10 random FOVs each)
REM ============================================================

set CACHE_ROOT=\\192.168.0.73\Papaya13\Lilian\merfish_sqi_cache
set OUT_ROOT=output\auc_comparison

set COND3=mouse_6OHDA:\\192.168.0.73\Papaya13\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set11
set COND2=human_control:\\192.168.0.116\durian3\Lilian\022425_FTD_smFISH_MBP_NRGN\coverslip1_controls\MBP_NRGN_set5
set COND1=human_FTD:\\192.168.0.116\durian3\Lilian\022425_FTD_smFISH_MBP_NRGN\coverslip2_FTD_group1\MBP_NRGN_set2

python scripts\run_auc_comparison.py ^
    --conditions "%COND1%" "%COND2%" "%COND3%" ^
    --cache_root "%CACHE_ROOT%" ^
    --out_root   "%OUT_ROOT%" ^
    --n_fovs 10 ^
    --seed 42

echo.
echo ============================================================
echo Done. Results in %OUT_ROOT%\
echo   - auc_comparison.png  (box plot)
echo   - auc_comparison.json (raw numbers)
echo ============================================================
pause
