# Content Integrity Guidelines for Paper Modifications

## ⚠️ CRITICAL: Preventing Content Loss

This document provides mandatory guidelines to prevent content loss when modifying academic papers using AI assistance.

## Pre-Modification Checklist

### 1. **ALWAYS Create Backups**
```powershell
# Before ANY modification, create a timestamped backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item paper.tex "paper.tex.backup_$timestamp"
```

### 2. **Document Baseline Metrics**
```powershell
# Capture baseline statistics BEFORE modification
$baseline = @{
    Lines = (Get-Content paper.tex).Count
    Sections = (Select-String -Path paper.tex -Pattern '\\section{').Matches.Count
    Subsections = (Select-String -Path paper.tex -Pattern '\\subsection{').Matches.Count
    Theorems = (Select-String -Path paper.tex -Pattern '\\begin{theorem}').Matches.Count
    Figures = (Select-String -Path paper.tex -Pattern '\\begin{figure}').Matches.Count
    Tables = (Select-String -Path paper.tex -Pattern '\\begin{table}').Matches.Count
    Equations = (Select-String -Path paper.tex -Pattern '\\begin{equation}').Matches.Count
    Algorithms = (Select-String -Path paper.tex -Pattern '\\begin{algorithm}').Matches.Count
}
$baseline | ConvertTo-Json | Out-File "baseline_metrics_$timestamp.json"
```

### 3. **Use Git for Version Control**
```powershell
# Commit current state before modifications
git add paper.tex
git commit -m "Baseline before modifications - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
```

## During Modification

### 1. **Apply Changes Incrementally**
- ❌ **NEVER** apply large diffs all at once
- ✅ **DO** apply changes section by section
- ✅ **DO** verify each change immediately after applying

### 2. **Use Safe Replacement Methods**
When using `replace_string_in_file`:
- Include 3-5 lines of context BEFORE the target
- Include 3-5 lines of context AFTER the target
- Use exact string matching (including whitespace)
- Verify the replacement location is unique

### 3. **Monitor for Warning Signs**
```powershell
# After each change, check line count
$currentLines = (Get-Content paper.tex).Count
if ($currentLines -lt ($baseline.Lines * 0.95)) {
    Write-Warning "⚠️ Line count dropped by >5%! Review changes immediately!"
}
```

## Post-Modification Verification

### 1. **Comprehensive Content Check**
Run this verification script after ANY modification:

```powershell
# Save this as: verify_content_integrity.ps1
param(
    [string]$BaselineFile = "baseline_metrics_*.json"
)

Write-Host "=== CONTENT INTEGRITY VERIFICATION ===" -ForegroundColor Cyan
Write-Host ""

# Load baseline
$baseline = Get-Content (Get-Item $BaselineFile | Sort-Object LastWriteTime -Descending | Select-Object -First 1) | ConvertFrom-Json

# Current metrics
$current = @{
    Lines = (Get-Content paper.tex).Count
    Sections = (Select-String -Path paper.tex -Pattern '\\section{').Matches.Count
    Subsections = (Select-String -Path paper.tex -Pattern '\\subsection{').Matches.Count
    Theorems = (Select-String -Path paper.tex -Pattern '\\begin{theorem}').Matches.Count
    Figures = (Select-String -Path paper.tex -Pattern '\\begin{figure}').Matches.Count
    Tables = (Select-String -Path paper.tex -Pattern '\\begin{table}').Matches.Count
    Equations = (Select-String -Path paper.tex -Pattern '\\begin{equation}').Matches.Count
    Algorithms = (Select-String -Path paper.tex -Pattern '\\begin{algorithm}').Matches.Count
}

# Compare and report
$issues = @()
foreach ($key in $baseline.PSObject.Properties.Name) {
    $baseVal = [int]$baseline.$key
    $currVal = [int]$current.$key
    $diff = $currVal - $baseVal
    $pctChange = if ($baseVal -gt 0) { ($diff / $baseVal) * 100 } else { 0 }
    
    $status = if ($diff -eq 0) { "✅" } 
              elseif ($diff -gt 0) { "➕" } 
              else { "⚠️" }
    
    Write-Host "$status $key`: $baseVal → $currVal ($('{0:+0;-0;0}' -f $diff), $('{0:+0.0;-0.0;0.0}' -f $pctChange)%)"
    
    # Flag significant losses
    if ($pctChange -lt -10) {
        $issues += "⚠️ CRITICAL: $key dropped by $([Math]::Abs([Math]::Round($pctChange, 1)))%"
    }
}

Write-Host ""
if ($issues.Count -gt 0) {
    Write-Host "=== ⚠️ ISSUES DETECTED ===" -ForegroundColor Red
    $issues | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    Write-Host ""
    Write-Host "🔴 RECOMMENDATION: Review changes and consider rollback!" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ Content integrity verified - no significant losses detected" -ForegroundColor Green
    exit 0
}
```

### 2. **Verify LaTeX Compilation**
```powershell
# Ensure the document still compiles
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PDF compilation successful"
} else {
    Write-Host "❌ PDF compilation failed - syntax errors introduced!"
    exit 1
}
```

### 3. **Visual Diff Check**
```powershell
# Compare with backup
git diff paper.tex | Out-File "changes_$(Get-Date -Format 'yyyyMMdd_HHmmss').diff"
Write-Host "📄 Diff saved for review"
```

### 4. **Key Section Verification**
Create a section checklist for your specific paper:

```powershell
# Verify all critical sections are present
$criticalSections = @(
    "\\section{Introduction",
    "\\section{Related Work",
    "\\section.*Formalism",
    "\\section.*Results",
    "\\section.*Discussion",
    "\\section.*Conclusion",
    "\\bibliography{"
)

$missing = @()
foreach ($section in $criticalSections) {
    if (-not (Select-String -Path paper.tex -Pattern $section -Quiet)) {
        $missing += $section
    }
}

if ($missing.Count -gt 0) {
    Write-Host "⚠️ MISSING CRITICAL SECTIONS:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
} else {
    Write-Host "✅ All critical sections present"
}
```

## Recovery Procedures

### If Content Loss is Detected:

1. **Immediate Rollback**
```powershell
# Restore from most recent backup
$latestBackup = Get-ChildItem "paper.tex.backup_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Copy-Item $latestBackup.FullName paper.tex -Force
Write-Host "✅ Restored from: $($latestBackup.Name)"
```

2. **Git Restore**
```powershell
# Restore from git
git checkout HEAD -- paper.tex
# or restore to specific commit
git checkout <commit-hash> -- paper.tex
```

3. **Manual Recovery**
- Open backup file and current file side-by-side
- Use a diff tool (e.g., WinMerge, Beyond Compare, VSCode diff)
- Manually merge missing content

## Best Practices for AI-Assisted Modifications

### ✅ DO:
1. **Apply diffs incrementally** - one section at a time
2. **Verify after each change** - check line counts and key sections
3. **Use version control** - commit frequently
4. **Keep multiple backups** - timestamped copies
5. **Test compilation** - after each major change
6. **Review diffs manually** - before accepting changes
7. **Use specific context** - include enough surrounding text in replacements
8. **Document changes** - keep a changelog

### ❌ DON'T:
1. **Don't skip backups** - EVER
2. **Don't apply large diffs blindly** - always review
3. **Don't ignore warnings** - investigate line count drops
4. **Don't work without version control** - git is your safety net
5. **Don't skip verification** - always run integrity checks
6. **Don't use vague search patterns** - be specific in replacements
7. **Don't assume success** - verify every change
8. **Don't forget to compile** - catch LaTeX errors early

## Automated Safeguards

### Pre-Change Hook
Create a script that runs BEFORE any modification:

```powershell
# pre_modify.ps1
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 1. Create backup
Copy-Item paper.tex "paper.tex.backup_$timestamp"

# 2. Save baseline
@{
    Timestamp = $timestamp
    Lines = (Get-Content paper.tex).Count
    Sections = (Select-String -Path paper.tex -Pattern '\\section{').Matches.Count
    Filesize = (Get-Item paper.tex).Length
} | ConvertTo-Json | Out-File "baseline_$timestamp.json"

# 3. Git commit
git add paper.tex
git commit -m "Auto-backup before modification - $timestamp" | Out-Null

Write-Host "✅ Pre-modification safeguards complete" -ForegroundColor Green
Write-Host "   Backup: paper.tex.backup_$timestamp"
Write-Host "   Baseline: baseline_$timestamp.json"
```

### Post-Change Hook
Create a script that runs AFTER any modification:

```powershell
# post_modify.ps1
param(
    [switch]$Force
)

# 1. Run integrity check
.\verify_content_integrity.ps1

if ($LASTEXITCODE -ne 0 -and -not $Force) {
    Write-Host "⚠️ Integrity check failed! Use -Force to proceed anyway." -ForegroundColor Red
    exit 1
}

# 2. Test compilation
Write-Host "Testing LaTeX compilation..." -ForegroundColor Cyan
pdflatex paper.tex 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ LaTeX compilation failed!" -ForegroundColor Red
    exit 1
}

# 3. Git commit changes
git add paper.tex
git commit -m "Modifications applied - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

Write-Host "✅ Post-modification verification complete" -ForegroundColor Green
```

## Paper-Specific Configuration

Create a `paper_config.json` for each paper:

```json
{
  "critical_sections": [
    "\\\\section{Introduction",
    "\\\\section.*Conclusion",
    "\\\\bibliography{"
  ],
  "min_line_count": 1000,
  "min_section_count": 5,
  "required_labels": [
    "\\\\label{sec:intro}",
    "\\\\label{sec:methods}",
    "\\\\label{sec:results}"
  ],
  "backup_dir": "backups/",
  "alert_on_loss_percent": 5
}
```

## Emergency Contact List

If you experience catastrophic content loss:

1. **STOP immediately** - don't make more changes
2. **Check backups** - you have timestamped backups, right?
3. **Check git history** - `git log --oneline`
4. **Check LaTeX auxiliary files** - sometimes content can be recovered
5. **Restore from last known good state**

## Summary Checklist

Before ANY modification:
- [ ] Create timestamped backup
- [ ] Save baseline metrics
- [ ] Commit to git
- [ ] Verify paper compiles

During modification:
- [ ] Apply changes incrementally
- [ ] Verify each change
- [ ] Monitor line counts

After modification:
- [ ] Run integrity verification script
- [ ] Test LaTeX compilation
- [ ] Review git diff
- [ ] Verify all critical sections present
- [ ] Commit changes to git

---

**Remember: Prevention is better than recovery. Always backup, always verify!**
