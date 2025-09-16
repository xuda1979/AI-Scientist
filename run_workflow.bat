@echo off
cd /d "C:\Users\Lenovo\software\simple-sciresearch-workflow"
C:\Users\Lenovo\miniconda3\python.exe sciresearch_workflow_refactored.py --modify-existing --output-dir output\ag-qec --model gpt-5 --max-iterations 1 --enable-pdf-review --check-references --validate-figures
