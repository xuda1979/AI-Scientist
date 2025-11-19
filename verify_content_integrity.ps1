# Content Integrity Verification Script
# Run this AFTER any paper modification to ensure no content was lost

param(
    [string]$PaperPath = "paper.tex",
    [string]$BaselinePattern = "baseline_*.json"
)

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     CONTENT INTEGRITY VERIFICATION                   ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if paper exists
if (-not (Test-Path $PaperPath)) {
    Write-Host "❌ ERROR: $PaperPath not found!" -ForegroundColor Red
    exit 1
}

# Find most recent baseline
$baselineFiles = Get-Item $BaselinePattern -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($baselineFiles) {
    $baselineFile = $baselineFiles | Select-Object -First 1
    Write-Host "📊 Comparing against baseline: $($baselineFile.Name)" -ForegroundColor Gray
    $baseline = Get-Content $baselineFile | ConvertFrom-Json
} else {
    Write-Host "⚠️  No baseline found - creating new baseline..." -ForegroundColor Yellow
    $baseline = $null
}

# Collect current metrics
Write-Host "📈 Collecting current metrics..." -ForegroundColor Gray
$current = @{
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Lines = (Get-Content $PaperPath).Count
    Sections = (Select-String -Path $PaperPath -Pattern '\\section\{' -AllMatches).Matches.Count
    Subsections = (Select-String -Path $PaperPath -Pattern '\\subsection\{' -AllMatches).Matches.Count
    Subsubsections = (Select-String -Path $PaperPath -Pattern '\\subsubsection\{' -AllMatches).Matches.Count
    Theorems = (Select-String -Path $PaperPath -Pattern '\\begin\{theorem\}' -AllMatches).Matches.Count
    Lemmas = (Select-String -Path $PaperPath -Pattern '\\begin\{lemma\}' -AllMatches).Matches.Count
    Propositions = (Select-String -Path $PaperPath -Pattern '\\begin\{proposition\}' -AllMatches).Matches.Count
    Corollaries = (Select-String -Path $PaperPath -Pattern '\\begin\{corollary\}' -AllMatches).Matches.Count
    Figures = (Select-String -Path $PaperPath -Pattern '\\begin\{figure\}' -AllMatches).Matches.Count
    Tables = (Select-String -Path $PaperPath -Pattern '\\begin\{table\}' -AllMatches).Matches.Count
    Equations = (Select-String -Path $PaperPath -Pattern '\\begin\{equation\}' -AllMatches).Matches.Count
    Algorithms = (Select-String -Path $PaperPath -Pattern '\\begin\{algorithm\}' -AllMatches).Matches.Count
    Citations = (Select-String -Path $PaperPath -Pattern '\\cite\{' -AllMatches).Matches.Count
    References = (Select-String -Path $PaperPath -Pattern '\\ref\{' -AllMatches).Matches.Count
    Labels = (Select-String -Path $PaperPath -Pattern '\\label\{' -AllMatches).Matches.Count
    FileSize = (Get-Item $PaperPath).Length
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

# Display comparison
$issues = @()
$warnings = @()

foreach ($key in ($current.Keys | Sort-Object)) {
    if ($key -eq "Timestamp") { continue }
    
    $currVal = [int]$current[$key]
    
    if ($baseline -and $baseline.PSObject.Properties.Name -contains $key) {
        $baseVal = [int]$baseline.$key
        $diff = $currVal - $baseVal
        $pctChange = if ($baseVal -gt 0) { ($diff / $baseVal) * 100 } else { 0 }
        
        $status = "  "
        $color = "White"
        
        if ($diff -eq 0) { 
            $status = "✅"
            $color = "Green"
        } elseif ($diff -gt 0) { 
            $status = "➕"
            $color = "Cyan"
        } else { 
            $status = "⚠️ "
            $color = "Yellow"
            
            # Critical loss detection
            if ($pctChange -lt -10) {
                $issues += "$key dropped by $([Math]::Abs([Math]::Round($pctChange, 1)))%"
                $color = "Red"
            } elseif ($pctChange -lt -5) {
                $warnings += "$key dropped by $([Math]::Abs([Math]::Round($pctChange, 1)))%"
            }
        }
        
        $diffStr = "{0,+5;-5;    0}" -f $diff
        $pctStr = "{0,+6:0.0;-6:0.0;  0.0}%" -f $pctChange
        Write-Host "$status " -NoNewline -ForegroundColor $color
        Write-Host ("{0,-18}" -f $key) -NoNewline
        Write-Host ": $baseVal → $currVal " -NoNewline
        Write-Host "($diffStr, $pctStr)" -ForegroundColor $color
    } else {
        # No baseline to compare
        Write-Host "ℹ️  {0,-18}: {1}" -f $key, $currVal -ForegroundColor Gray
    }
}

Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Critical sections check
Write-Host "🔍 Verifying critical sections..." -ForegroundColor Cyan
$criticalSections = @(
    @{ Pattern = '\\begin\{document\}'; Name = "Document Begin" },
    @{ Pattern = '\\section\{.*Introduction'; Name = "Introduction Section" },
    @{ Pattern = '\\section\{.*[Cc]onclusion'; Name = "Conclusion Section" },
    @{ Pattern = '\\bibliography\{|\\begin\{thebibliography\}'; Name = "Bibliography" },
    @{ Pattern = '\\end\{document\}'; Name = "Document End" }
)

$missingCritical = @()
foreach ($section in $criticalSections) {
    if (Select-String -Path $PaperPath -Pattern $section.Pattern -Quiet) {
        Write-Host "   ✅ $($section.Name)" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $($section.Name) - MISSING!" -ForegroundColor Red
        $missingCritical += $section.Name
    }
}

Write-Host ""

# Final verdict
$exitCode = 0

if ($missingCritical.Count -gt 0) {
    Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║  🔴 CRITICAL FAILURE - MISSING ESSENTIAL SECTIONS    ║" -ForegroundColor Red
    Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    Write-Host "Missing sections:" -ForegroundColor Red
    $missingCritical | ForEach-Object { Write-Host "  ❌ $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "🔴 IMMEDIATE ACTION REQUIRED: Restore from backup!" -ForegroundColor Red
    $exitCode = 2
}
elseif ($issues.Count -gt 0) {
    Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║  ⚠️  CRITICAL ISSUES DETECTED                        ║" -ForegroundColor Red
    Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    Write-Host "Critical content losses detected:" -ForegroundColor Red
    $issues | ForEach-Object { Write-Host "  ⚠️  $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "🔴 RECOMMENDATION: Review changes and consider rollback!" -ForegroundColor Red
    $exitCode = 1
}
elseif ($warnings.Count -gt 0) {
    Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║  ⚠️  WARNINGS DETECTED                               ║" -ForegroundColor Yellow
    Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Minor content changes detected:" -ForegroundColor Yellow
    $warnings | ForEach-Object { Write-Host "  ⚠️  $_" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "💡 Review changes to ensure they are intentional" -ForegroundColor Yellow
    $exitCode = 0
}
else {
    Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║  ✅ CONTENT INTEGRITY VERIFIED                       ║" -ForegroundColor Green
    Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "✅ No significant content losses detected" -ForegroundColor Green
    Write-Host "✅ All critical sections present" -ForegroundColor Green
    $exitCode = 0
}

Write-Host ""

# Save current as new baseline if requested
if ($env:SAVE_BASELINE -eq "1" -or $baseline -eq $null) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $baselineFile = "baseline_$timestamp.json"
    $current | ConvertTo-Json | Out-File $baselineFile
    Write-Host "💾 New baseline saved: $baselineFile" -ForegroundColor Cyan
}

exit $exitCode
