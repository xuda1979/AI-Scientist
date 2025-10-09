@echo off
setlocal ENABLEEXTENSIONS

rem Simple helper script to run the workflow on Windows without referencing a specific paper.
rem Usage: run_workflow.bat "Topic" "Field" "Research question" [additional workflow args]

if "%~3"=="" (
    echo Usage: %~nx0 "Topic" "Field" "Research question" [additional workflow args]
    exit /b 1
)

set "TOPIC=%~1"
set "FIELD=%~2"
set "QUESTION=%~3"

shift
shift
shift

pushd "%~dp0"
python "%~dp0sciresearch_workflow.py" --topic "%TOPIC%" --field "%FIELD%" --question "%QUESTION%" %*
popd

endlocal
