# Pre-Modification Baseline Creation Script
# Run this BEFORE any paper modification to create a baseline snapshot

param(
    [string]$PaperPath = "paper.tex"
)

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     PRE-MODIFICATION BASELINE CREATION               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if paper exists
if (-not (Test-Path $PaperPath)) {
    Write-Host "❌ ERROR: $PaperPath not found!" -ForegroundColor Red
    exit 1
}

# Create timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 1. Create backup directory
$backupDir = "backups"
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir | Out-Null
}

# 2. Create full file backup
$backupPath = Join-Path $backupDir "paper_$timestamp.tex"
Copy-Item $PaperPath $backupPath
Write-Host "📦 Backup created: $backupPath" -ForegroundColor Green

# 3. Calculate file hash
$hash = Get-FileHash $PaperPath -Algorithm SHA256
Write-Host "🔐 File hash: $($hash.Hash)" -ForegroundColor Gray

# 4. Collect detailed metrics
Write-Host "📊 Collecting content metrics..." -ForegroundColor Gray

$baseline = @{
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    FilePath = $PaperPath
    BackupPath = $backupPath
    Hash = $hash.Hash
    FileSize = (Get-Item $PaperPath).Length
    Lines = (Get-Content $PaperPath).Count
    
    # Structure counts
    Sections = (Select-String -Path $PaperPath -Pattern '\\section\{' -AllMatches).Matches.Count
    Subsections = (Select-String -Path $PaperPath -Pattern '\\subsection\{' -AllMatches).Matches.Count
    Subsubsections = (Select-String -Path $PaperPath -Pattern '\\subsubsection\{' -AllMatches).Matches.Count
    
    # Math content
    Theorems = (Select-String -Path $PaperPath -Pattern '\\begin\{theorem\}' -AllMatches).Matches.Count
    Lemmas = (Select-String -Path $PaperPath -Pattern '\\begin\{lemma\}' -AllMatches).Matches.Count
    Propositions = (Select-String -Path $PaperPath -Pattern '\\begin\{proposition\}' -AllMatches).Matches.Count
    Corollaries = (Select-String -Path $PaperPath -Pattern '\\begin\{corollary\}' -AllMatches).Matches.Count
    Definitions = (Select-String -Path $PaperPath -Pattern '\\begin\{definition\}' -AllMatches).Matches.Count
    Proofs = (Select-String -Path $PaperPath -Pattern '\\begin\{proof\}' -AllMatches).Matches.Count
    
    # Visual content
    Figures = (Select-String -Path $PaperPath -Pattern '\\begin\{figure\}' -AllMatches).Matches.Count
    Tables = (Select-String -Path $PaperPath -Pattern '\\begin\{table\}' -AllMatches).Matches.Count
    
    # Equations
    Equations = (Select-String -Path $PaperPath -Pattern '\\begin\{equation\}' -AllMatches).Matches.Count
    Align = (Select-String -Path $PaperPath -Pattern '\\begin\{align' -AllMatches).Matches.Count
    
    # Algorithms
    Algorithms = (Select-String -Path $PaperPath -Pattern '\\begin\{algorithm\}' -AllMatches).Matches.Count
    
    # References
    Citations = (Select-String -Path $PaperPath -Pattern '\\cite\{' -AllMatches).Matches.Count
    References = (Select-String -Path $PaperPath -Pattern '\\ref\{' -AllMatches).Matches.Count
    Labels = (Select-String -Path $PaperPath -Pattern '\\label\{' -AllMatches).Matches.Count
}

# 5. Extract section titles for detailed comparison
$sectionMatches = Select-String -Path $PaperPath -Pattern '\\section\{([^}]+)\}' -AllMatches
$baseline.SectionTitles = @($sectionMatches.Matches | ForEach-Object { $_.Groups[1].Value })

# 6. Save baseline as JSON
$baselineFile = "baseline_$timestamp.json"
$baseline | ConvertTo-Json | Out-File $baselineFile
Write-Host "💾 Baseline saved: $baselineFile" -ForegroundColor Green

# 7. Display summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  CONTENT SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Lines:          $($baseline.Lines)" -ForegroundColor White
Write-Host "  File Size:      $($baseline.FileSize) bytes" -ForegroundColor White
Write-Host ""
Write-Host "  Sections:       $($baseline.Sections)" -ForegroundColor Cyan
Write-Host "  Subsections:    $($baseline.Subsections)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Theorems:       $($baseline.Theorems)" -ForegroundColor Yellow
Write-Host "  Lemmas:         $($baseline.Lemmas)" -ForegroundColor Yellow
Write-Host "  Propositions:   $($baseline.Propositions)" -ForegroundColor Yellow
Write-Host "  Definitions:    $($baseline.Definitions)" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Figures:        $($baseline.Figures)" -ForegroundColor Green
Write-Host "  Tables:         $($baseline.Tables)" -ForegroundColor Green
Write-Host "  Equations:      $($baseline.Equations)" -ForegroundColor Green
Write-Host "  Algorithms:     $($baseline.Algorithms)" -ForegroundColor Green
Write-Host ""
Write-Host "  Citations:      $($baseline.Citations)" -ForegroundColor Magenta
Write-Host "  References:     $($baseline.References)" -ForegroundColor Magenta
Write-Host "  Labels:         $($baseline.Labels)" -ForegroundColor Magenta
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

# 8. Check if Git repository exists
if (Test-Path ".git") {
    Write-Host ""
    Write-Host "📌 Git repository detected" -ForegroundColor Gray
    
    # Get current git status
    $gitStatus = git status --porcelain $PaperPath 2>$null
    if ($gitStatus) {
        Write-Host "⚠️  WARNING: File has uncommitted changes" -ForegroundColor Yellow
        Write-Host "   Consider committing before modification" -ForegroundColor Yellow
    } else {
        Write-Host "✅ File is clean in Git" -ForegroundColor Green
    }
    
    # Show last commit affecting this file
    $lastCommit = git log -1 --format="%h %s (%ar)" -- $PaperPath 2>$null
    if ($lastCommit) {
        Write-Host "   Last change: $lastCommit" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅ BASELINE CREATED - READY FOR MODIFICATION        ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Make your modifications to $PaperPath" -ForegroundColor White
Write-Host "   2. Run: .\verify_content_integrity.ps1" -ForegroundColor White
Write-Host "   3. If issues detected, restore from: $backupPath" -ForegroundColor White
Write-Host ""

# Output baseline filename for piping to verification script
return $baselineFile
