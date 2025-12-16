# Physical Review D Format Conversion Summary

This document summarizes the changes made to convert `paper.tex` from a standard article format to Physical Review D (PRD) format using REVTeX 4.2.

## Major Changes

### 1. Document Class
**Before:**
```latex
\documentclass[11pt,a4paper]{article}
```

**After:**
```latex
\documentclass[aps,prd,reprint,superscriptaddress,nofootinbib,longbibliography,floatfix]{revtex4-2}
```

**Options explained:**
- `reprint`: APS single-column submission format
- `aps`: American Physical Society style
- `prd`: Physical Review D specific formatting
- `superscriptaddress`: Affiliations as superscripts
- `nofootinbib`: Keep references in bibliography section
- `longbibliography`: Ensure full author lists per APS style
- `floatfix`: Improve float placement within REVTeX constraints

### 2. Removed Packages
- `\usepackage[margin=1.2in]{geometry}` - REVTeX handles page layout
- `\usepackage{setspace}` with `\setstretch{1.6}` - REVTeX controls spacing
- `\usepackage[numbers,sort&compress]{natbib}` - REVTeX includes natbib
- Direct `\usepackage{amsmath}` - REVTeX includes this

> **Note:** We now load `hyperref` explicitly at the end of the preamble with APS-safe color options to avoid the need for manual fallbacks such as `\texorpdfstring`.

### 3. Author and Affiliation Format
**Before:**
```latex
\author{Da Xu\\
\small China Mobile Research Institute, Beijing, P. R. China\\
\small \texttt{xudayj@chinamobile.com}}
```

**After:**
```latex
\author{Da Xu}
\affiliation{China Mobile Research Institute, Beijing, P. R. China}
\email{xudayj@chinamobile.com}
```

### 4. Added PACS Codes
Added after abstract:
```latex
\pacs{04.70.Dy, 04.62.+v, 03.67.Mn, 89.70.Cf}
```

**PACS codes meaning:**
- 04.70.Dy: Quantum aspects of black holes, evaporation, thermodynamics
- 04.62.+v: Quantum fields in curved spacetime
- 03.67.Mn: Entanglement measures, witnesses, and other characterizations
- 89.70.Cf: Entropy and other measures of information

### 5. Acknowledgments Format
**Before:**
```latex
\section*{Acknowledgments}
I would like to express...
```

**After:**
```latex
\begin{acknowledgments}
I would like to express...
\end{acknowledgments}
```

### 6. Bibliography Style
**Before:**
```latex
\bibliographystyle{unsrt}
```

**After:**
```latex
\bibliographystyle{apsrev4-2}
```

### 7. Removed Sections
- Removed "Conflict of Interest" section (not standard for PRD)

## Package Compatibility Notes

1. **caption package**: Already commented out (REVTeX handles captions)
2. **cleveref**: Kept but note that REVTeX has its own cross-referencing
3. **microtype**: Kept for improved typography
4. **subcaption**: Removed to avoid conflicts flagged by APS/REVTeX (native figure environments preferred)
5. **algorithm packages**: Kept for algorithm environments

## Recommendations for Final Submission

1. **Ensure the `showkeys` option remains disabled** in the documentclass (already removed)
2. **Check all cross-references** compile correctly with REVTeX
3. **Verify all figures** display properly in single-column format
4. **Review bibliography entries** to ensure they follow APS style
5. **Check table widths** - may need adjustment for PRD column width
6. **Consider two-column format**: Change `reprint` to `twocolumn` if needed

## Testing the Compilation

Compile with:
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or use latexmk:
```bash
latexmk -pdf paper.tex
```

## Additional Notes

- The paper retains most custom environments (boxedresult, theorem environments, etc.)
- TikZ figures and pgfplots should work with REVTeX
- All mathematical notation remains unchanged
- Custom commands and macros are preserved

## Follow-up alignment (November 2025)

- Added `longbibliography` and `floatfix` options plus APS-recommended keywords support before `\maketitle`.
- Reordered `\email`/`\affiliation` declarations and moved `\maketitle` below the abstract, PACS, and keywords to match the APS sample file ordering.
- Removed `subcaption` and duplicate `tabularx` loads; added an explicit, last-loaded `hyperref` block with colorlinks.
- Documented the new compliance steps here to keep the PRD tooling audit trail current.

## Physical Review D Specific Requirements Met

✓ REVTeX 4.2 document class
✓ APS and PRD options specified (longbibliography + floatfix for APS compliance)
✓ PACS codes included
✓ Keywords defined via `\keywords{}` before `\maketitle`
✓ Author affiliations in proper format
✓ Acknowledgments in dedicated environment
✓ Bibliography style apsrev4-2
✓ Single-column reprint format for submission
