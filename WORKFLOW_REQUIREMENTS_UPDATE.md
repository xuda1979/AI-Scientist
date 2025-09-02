# Workflow Requirements Update

## Summary of Changes

The `sciresearch_workflow.py` has been updated to include three critical requirements for the revision process:

### 1. Single LaTeX File Requirement
- **Prompt Update**: All prompts now specify that the paper must be contained in ONE LaTeX file
- **Validation**: Added checks for `\input{}` and `\include{}` commands that would violate single-file requirement
- **Error Detection**: Quality validation function now flags papers that use external file inclusion

### 2. Embedded References Requirement
- **Initial Draft**: Updated `_initial_draft_prompt()` to require embedded references using `\begin{filecontents*}{refs.bib}...\end{filecontents*}`
- **Revision Prompt**: Updated `_revise_prompt()` to include detailed examples of bibliography integration options
- **Validation**: Added checks to ensure references are embedded (either filecontents or thebibliography)
- **Quality Control**: Flags papers that reference external .bib files without embedding them

### 3. LaTeX Compilation with Error Checking
- **New Function**: Added `_compile_latex_and_get_errors()` that:
  - Runs `pdflatex -interaction=nonstopmode` on the paper
  - Checks if PDF was successfully generated
  - Extracts last 20 lines of the .log file for error analysis
  - Returns success status and error details
- **Workflow Integration**: Added compilation check in main review loop:
  - Compiles LaTeX after each iteration
  - Sends compilation errors to LLM if compilation fails
  - Only accepts paper if both editorial decision is YES AND LaTeX compiles successfully
- **Error Handling**: LLM receives detailed error logs to fix compilation issues

## Updated Workflow Process

### For Each Iteration:
1. Run simulation to get current results
2. **Compile LaTeX with pdflatex -interaction=nonstopmode**
3. **Extract last 20 lines of .log file**
4. Validate research quality (including new requirements)
5. Send paper + simulation + **LaTeX errors (if any)** to reviewer LLM
6. Get editorial decision
7. **Only accept if decision is YES AND LaTeX compiles successfully**
8. If not accepted, send paper + simulation + review + **LaTeX errors** to revision LLM
9. Revision LLM receives detailed prompts about:
   - Single file requirement
   - Embedded references requirement
   - LaTeX compilation error fixes

## New Prompt Features

### Initial Draft Prompt
- Requires `\begin{filecontents*}{refs.bib}...\end{filecontents*}` at top of file
- Specifies minimum 15-20 references
- Provides example structure for self-contained paper

### Review Prompt
- Added "MANDATORY REQUIREMENTS" section checking all 3 requirements
- Marks papers as needing major revision if requirements are violated
- Enhanced formatting criteria for LaTeX compilation success

### Revision Prompt
- Added "CRITICAL REQUIREMENTS" section with detailed enforcement
- Provides two bibliography integration options (filecontents vs thebibliography)
- Includes LaTeX error log analysis when compilation fails
- Enhanced size constraint examples to prevent formatting issues

## Validation Enhancements

### Quality Validation Function
- Checks for `\input{}` and `\include{}` usage (violation of single file rule)
- Validates bibliography embedding (filecontents vs external files)
- Ensures minimum reference count
- Maintains existing figure/table validation

### Compilation Validation
- Timeout protection (120 seconds)
- PDF generation verification
- Detailed error log extraction
- Graceful handling of missing pdflatex

## Benefits

1. **Consistency**: Every paper will be a single, self-contained LaTeX file
2. **Portability**: No external dependencies or missing file issues
3. **Quality Assurance**: Compilation checking ensures papers actually work
4. **Error Debugging**: LLM receives specific LaTeX errors to fix them intelligently
5. **Automated Validation**: Systematic checking of all requirements before acceptance

## Usage

The workflow maintains the same command-line interface. The new requirements are automatically enforced during the revision process:

```bash
python sciresearch_workflow.py --topic "Topic" --field "Field" --question "Question" --modify-existing
```

The workflow will now:
- Generate papers with embedded references
- Check LaTeX compilation on every iteration
- Send compilation errors to the LLM for fixing
- Only accept papers that meet all requirements AND compile successfully
