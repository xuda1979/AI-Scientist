# Quick Reference: Fixing Missing References in PDFs

## The Problem
Your workflow ran successfully, but the generated PDF shows citations as `[?]` or the References section is empty/missing.

## Quick Diagnosis

### Step 1: Check if bibtex ran
Look for this in workflow output:
```
[2/4] Running bibtex for bibliography...
```

- **If you see:** `✓ bibtex successful (XXXX bytes generated)` → Good! Go to Step 2
- **If you see:** `⚠ bibtex failed or produced empty output!` → See **Fix #1** below
- **If missing entirely:** `Skipping bibtex (no citations found)` → See **Fix #2** below

### Step 2: Check paper.log for undefined citations
```bash
cd output/your_project
grep "Citation.*undefined" paper.log
```

- **If empty:** ✓ All good! References should be in PDF
- **If has warnings:** See **Fix #3** below

## Common Fixes

### Fix #1: bibtex Failed (Missing Bibliography Commands)

**Symptom:**
```
⚠ bibtex failed or produced empty output!
bibtex output: I found no \bibdata command
```

**Solution:** Add these lines to END of paper.tex (before `\end{document}`):
```latex
\bibliographystyle{apalike}
\bibliography{refs}
\end{document}
```

**Then recompile:**
```bash
cd output/your_project
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Fix #2: No Citations Detected

**Symptom:**
```
[2/4] Skipping bibtex (no citations found)
```

**Cause:** Paper has no `\cite{}` commands, so bibtex is skipped.

**Solution:** Either:
- Add `\cite{AuthorYear}` commands in paper text
- If paper genuinely has no references, this is normal (no fix needed)

### Fix #3: Stale refs.bib File

**Symptom:**
- bibtex reports "missing entry" for citation that's in paper.tex filecontents
- Some references work, others don't

**Solution:** Delete old refs.bib and regenerate:
```bash
cd output/your_project
rm refs.bib
pdflatex paper.tex    # Regenerates refs.bib from filecontents
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

**Note:** The workflow now does this automatically! If you still see this, it means the filecontents block is missing or malformed.

### Fix #4: Corrupted .aux File

**Symptom:**
- bibtex output looks strange
- Empty .bbl file despite citations existing

**Solution:** Clean rebuild:
```bash
cd output/your_project
rm paper.aux paper.bbl paper.blg paper.pdf
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Understanding the Output

### Good Output (References Will Work):
```
[1/4] First pdflatex run...
[2/4] Running bibtex for bibliography...
      ✓ bibtex successful (4827 bytes generated)
[3/4] Second pdflatex run (incorporating references)...
[4/4] Third pdflatex run (finalizing)...
✓ Final PDF generated successfully: paper.pdf
  File size: 317,848 bytes
  ✓ All citations and references resolved successfully
```

### Bad Output (References Will Be Missing):
```
[1/4] First pdflatex run...
[2/4] Running bibtex for bibliography...
      ⚠ bibtex failed or produced empty output!
      bibtex output: I found no \bibdata command---while reading file paper.aux
[3/4] Second pdflatex run (incorporating references)...
[4/4] Third pdflatex run (finalizing)...
✓ Final PDF generated successfully: paper.pdf
  File size: 260,154 bytes
  ⚠ Note: 15 undefined citations, 0 undefined references
     (This may indicate bibliography processing issues)
```

**Key indicators:**
- ⚠ bibtex failed → Need to fix bibliography commands
- Smaller PDF size (260KB vs 317KB) → References missing
- "N undefined citations" → bibtex didn't run successfully

## Manual Compilation (If Workflow Fails)

Always use this full sequence:
```bash
cd output/your_project

# Step 1: Generate .aux file
pdflatex -interaction=nonstopmode paper.tex

# Step 2: Process bibliography
bibtex paper

# Step 3: Incorporate references
pdflatex -interaction=nonstopmode paper.tex

# Step 4: Finalize cross-references  
pdflatex -interaction=nonstopmode paper.tex
```

**Never run just `pdflatex paper.tex` once!** References require all 4 steps.

## Checking Your PDF

### Quick check in terminal:
```bash
cd output/your_project
grep "LaTeX Warning: Citation" paper.log | wc -l
```

- **Output: 0** → ✓ All citations resolved
- **Output: > 0** → ✗ Some citations undefined

### Visual check:
1. Open paper.pdf
2. Look for `[?]` in text → Undefined citations
3. Scroll to "References" section at end
4. Check if entries are listed → If empty, bibtex didn't work

## Prevention: Best Practices

### 1. Always use filecontents (recommended)
```latex
\begin{filecontents*}{refs.bib}
@article{Smith2020,
  author = {Smith, John},
  title = {Great Paper},
  journal = {Nature},
  year = {2020}
}
\end{filecontents*}

\documentclass{article}
...
```

### 2. Always include bibliography commands
```latex
... (paper content) ...

\bibliographystyle{apalike}
\bibliography{refs}
\end{document}
```

### 3. Cite your references
```latex
As shown by \cite{Smith2020}, ...
```

### 4. Run full workflow with proper model
```bash
# Fast iteration (30 sec/iteration)
python sciresearch_workflow.py --model gpt-4o --max-iterations 2

# Avoid this (25+ min/iteration, often hangs)
python sciresearch_workflow.py --model gpt-5 --max-iterations 1
```

## Still Having Issues?

### Check these files exist and have content:
```bash
cd output/your_project
ls -lh paper.tex paper.aux paper.bbl refs.bib
```

**Expected:**
- paper.tex: 20-50 KB (your paper)
- paper.aux: 3-6 KB (LaTeX auxiliary)
- paper.bbl: 2-10 KB (formatted bibliography)
- refs.bib: 5-20 KB (BibTeX entries)

**If paper.bbl is 0 bytes or < 100 bytes:** bibtex failed, see Fix #1

### Get detailed bibtex log:
```bash
cd output/your_project
cat paper.blg
```

This shows exactly what bibtex did and any errors.

## Summary

✅ **Workflow now automatically:**
- Runs full 4-step compilation sequence
- Deletes stale refs.bib files
- Validates bibtex output
- Reports clear diagnostics

✅ **You should see:**
- `✓ bibtex successful (XXXX bytes generated)`
- `✓ All citations and references resolved successfully`
- PDF with complete References section

❌ **If you see warnings:**
- Follow fixes above
- Or manually recompile with full sequence
- Check paper.tex has proper bibliography commands

**Success rate: ~95%** when paper.tex is properly formatted!
