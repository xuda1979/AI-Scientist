@echo off
REM Script to run AI-Scientist workflow to fix LaTeX errors in existing paper
REM This will:
REM 1. Detect current LaTeX compilation errors
REM 2. Send errors to the LLM
REM 3. LLM will fix the errors and improve the paper
REM 4. Iterate until paper compiles successfully

echo ================================================================================
echo AI-Scientist: Modify Existing Paper to Fix LaTeX Errors
echo ================================================================================
echo.
echo This will run the AI-Scientist workflow to:
echo   - Fix LaTeX compilation errors in output/black_hole/paper.tex
echo   - Ensure all plots render correctly with data
echo   - Improve paper quality based on validation checks
echo.
echo The LLM will receive the LaTeX error log and fix the issues.
echo.
echo ================================================================================
echo.

REM Activate conda environment if needed
REM call conda activate base

REM Run the workflow with error fixing instructions
python main.py --modify-existing ^
               --output-dir output ^
               --user-prompt "Fix all LaTeX compilation errors. The plots are showing empty because pgfplots cannot read the .dat files. Ensure all data files are properly embedded or referenced. All plots must show actual data, not be empty." ^
               --max-iterations 3 ^
               --model gpt-4o ^
               --request-timeout 600

echo.
echo ================================================================================
echo Workflow complete!
echo Check output/black_hole/paper.pdf for results
echo ================================================================================
pause
