# Git Diff Application Status

## Summary
The diff contains approximately 1,600 lines of changes across paper.tex and simulation.py. Due to the size, I'm applying changes in priority order.

## Changes Applied Successfully

### paper.tex
1. ✅ Added `\input{macros.tex}` after datatableExactComb loading
2. ✅ Updated PT-MPO dataset filenames:
   - `ptmpo_page_v6.dat` → `datatablePTMPOPageCurve.dat`
   - `ptmpo_scaling_v6.dat` → `datatablePTMPOscaling.dat`
   - `mps_error_v5.dat` → `datatablePTMPOerror.dat`
3. ✅ Added 18 new dataset declarations:
   - datatableEchoThresholds.dat
   - datatableDetectorSNR.dat
   - datatableConstantsLedger.dat
   - datatableDressingDilation.dat
   - datatableHydroOTOC.dat
   - datatableKcov.dat
   - datatableAnalogSidebands.dat
   - datatableAfterglow.dat
   - datatableEchoWaveform.dat
   - datatablePhysicalScales.dat
   - datatableObservability.dat
   - datatableEchoSystematics.dat
   - datatableChecksums.dat
   - datatablePetzRecovery.dat
   - datatableSchwarzianKernel.dat
   - datatableManifest.dat
   - datatableTruncationError.dat
   - datatableNullDist.dat

### simulation.py
1. ✅ Updated `_fmt_val()` to use scientific notation for small numbers (< 1e-3)
2. ✅ Added `write_table_tab()` function for tab-separated output

## Changes Still Needed (Priority Order)

### HIGH PRIORITY - Paper won't compile without these:
1. ❌ Create macros.tex file with dynamic values
2. ❌ Update simulation.py main() to generate all new datasets
3. ❌ Rewrite run_ablation_scenarios() to return raw_data for statistics
4. ❌ Add calculate_ablation_statistics() function
5. ❌ Add run_qec_simulation() function
6. ❌ Add pt_mpo_scaling_tables() function
7. ❌ Add run_exact_comb_simulation() function

### MEDIUM PRIORITY - Tables/Figures:
- Update Table 9 (Error Budget) formatting
- Update Table 12 & 13 (Ablation) to use dynamic data
- Update Table 15 (QEC) to use simulation data
- Add Figure: Dressing Dilation
- Add Figure: Hydro OTOC
- Add Figure: Afterglow Spectrum
- Add Figure: Echo Waveform
- Add Figure: Null Distribution
- Add Figure: Truncation Error
- Add Figure: Petz Recovery
- Add Figure: Kernel Comparison

### LOW PRIORITY - Theory sections:
- Add Remark on Geometric N
- Add Proposition P4' (QNEC)
- Add Remark on Symmetry Protection
- Update Kernel equations with rotation terms

## Next Steps
1. Create the 20+ new data generation functions in simulation.py
2. Update main() to call all new functions
3. Generate macros.tex file
4. Update table formatting in paper.tex to use dynamic data
5. Add new figures with TikZ plots

## Note
The complete diff would require:
- ~600 lines of new Python functions
- ~400 lines of table reformatting
- ~200 lines of new figures
- ~400 lines of theorem/remark additions

Total estimated: ~1600 lines of changes
