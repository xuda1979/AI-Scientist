# Paper Compilation Files

This directory contains only the essential files needed to compile `paper.tex`.

## Files Kept (58 total)

### LaTeX Source Files (6)
- `paper.tex` - Main LaTeX document
- `paper.bbl` - Compiled bibliography
- `references.bib` - Bibliography database
- `macros.tex` - Custom macros (loaded via \InputIfFileExists)
- `jheppub.sty` - JHEP journal style file
- `JHEP.bst` - JHEP bibliography style

### Data Files (48)
All `datatable*.dat` files loaded via `\pgfplotstableread` commands in the paper:
- datatableAblation.dat
- datatableAblationSig.dat
- datatableAfterglow.dat
- datatableAnalogSidebands.dat
- datatableArtifactMap.dat
- datatableChecksums.dat
- datatableConstantsLedger.dat
- datatableCVsummary.dat
- datatableDetectorSNR.dat
- datatableDressingDilation.dat
- datatableEchoExample_30Msun.dat
- datatableEchoSNR.dat
- datatableEchoSystematics.dat
- datatableEchoThresholds.dat
- datatableEchoTrials.dat
- datatableEchoWaveform.dat
- datatableEinsteinLangevin.dat
- datatableErrorBudget.dat
- datatableExactComb.dat
- datatableGreybodyErrorSweep.dat
- datatableGreybodyKbinToy.dat
- datatableGreybodySweep.dat
- datatableGtwo.dat
- datatableHeavyTail.dat
- datatableHydroOTOC.dat
- datatableKcov.dat
- datatableManifest.dat
- datatableMarkovBaseline.dat
- datatableMinimalCombRE.dat
- datatableNullDist.dat
- datatableObservability.dat
- datatablePagecurve.dat
- datatablePagecurveAblations.dat
- datatablePetzRecovery.dat
- datatablePhotonSphereLyap.dat
- datatablePhysicalScales.dat
- datatablePTMPO.dat
- datatablePTMPOCertificate.dat
- datatablePTMPOerror.dat
- datatablePTMPOPageCurve.dat
- datatablePTMPOscaling.dat
- datatableQEC.dat
- datatableQEIVariance.dat
- datatableQTPEgap.dat
- datatableRingdown3Pole.dat
- datatableSchwarzianKernel.dat
- datatableSoftFlux.dat
- datatableTruncationError.dat

### Figure Files (4 in fig/ directory)
- fig/comb_code_embedding.pdf
- fig/identifiability_true_vs_est_kernel.pdf
- fig/kk_residuals.pdf
- fig/kk_re_vs_hilbert.pdf

## Files Removed (98 files)

All non-essential files were removed including:
- Log files (*.log)
- Auxiliary LaTeX files (*.aux, *.out, *.toc, *.blg)
- Backup files (*.bak)
- Python scripts (*.py)
- Markdown documentation (*.md)
- CSV data files (*.csv)
- JSON metadata files (*.json)
- Alternative data versions (*_v4.dat, *_v5.dat except datatables)
- Build scripts (Makefile, *.sh)
- Output directories (backups/, code/, guardian/, output/, test_out/, updates/, __pycache__/)
- Archive files (*.zip, *.lzma)
- Text files (*.txt)

## To Compile

Run standard LaTeX commands:
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or use your preferred LaTeX editor/build system.
