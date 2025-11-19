# Multi-Layer Content Protection System

## 🛡️ Overview

This system provides **6 layers of defense** against content loss during AI-powered paper revisions.

## 📋 Protection Layers

### Layer 1: Content Guardian (Automatic)
- **What**: Validates every edit before applying
- **Protects Against**: 
  - File truncation (missing `\end{document}`)
  - Excessive content loss (>5% lines or size)
  - Structural damage (lost sections, bibliography)
- **Action**: Automatically blocks unsafe edits

### Layer 2: Checkpoint System (Automatic)
- **What**: Creates versioned backups before every change
- **Protects Against**: Irreversible mistakes
- **Action**: Can rollback to any checkpoint

### Layer 3: Content Protector (Existing)
- **What**: Word-count based validation
- **Protects Against**: Large-scale deletions
- **Action**: Requires user approval for risky changes

### Layer 4: Pre-Flight Check (Manual)
- **What**: Validates paper before starting revisions
- **Protects Against**: Starting with corrupted file
- **Action**: Warns if paper is already damaged

### Layer 5: Emergency Recovery (Manual)
- **What**: Quick restoration tool
- **Protects Against**: All types of corruption
- **Action**: Restores from multiple backup sources

### Layer 6: Integrity Monitor (Optional)
- **What**: Real-time file watching
- **Protects Against**: Silent corruption
- **Action**: Alerts and auto-recovers from truncation

## 🚀 Usage

### Before Running Revisions

**Step 1: Pre-Flight Check**
```bash
python preflight_check.py output/black_hole/paper.tex
```

This will:
- ✅ Verify paper integrity
- ✅ Create safety checkpoint
- ✅ Report any issues
- ✅ Tell you if it's safe to proceed

### During Revisions (Optional)

**Step 2: Start Integrity Monitor** (in separate terminal)
```bash
python monitor_content_integrity.py output/black_hole/paper.tex
```

This will:
- 📡 Watch for file changes
- 🚨 Alert on suspicious changes
- 🔄 Auto-recover from truncation

### Running Revisions

**Step 3: Run with Protection Enabled**
```bash
python main.py --modify-existing --output-dir ./output/black_hole/ --max-iterations 5 --model gemini-1.5-pro
```

**Protection is now automatic!** The Content Guardian will:
- Create checkpoint before each iteration
- Validate every proposed change
- Block destructive edits
- Automatically rollback if needed

### If Something Goes Wrong

**Emergency Recovery**
```bash
python emergency_recovery.py output/black_hole/paper.tex
```

This will:
- 🔍 Show all available recovery points
- 📋 Let you choose which to restore
- ✅ Verify recovery success

## 📊 Protection Thresholds

The Content Guardian enforces these limits:

| Check | Threshold | Action |
|-------|-----------|--------|
| Minimum lines | 2,000 | Block if below |
| Minimum size | 140 KB | Block if below |
| Max line loss | 5% | Block if exceeded |
| Max size loss | 5% | Block if exceeded |
| Missing `\end{document}` | N/A | Block always |
| Lost bibliography | N/A | Block always |
| Lost >2 sections | N/A | Block always |

## 🔍 Monitoring & Debugging

### Check Guardian Status
```python
from pathlib import Path
from utils.content_guardian import ContentGuardian

guardian = ContentGuardian(Path("output/black_hole"))
guardian.show_history()
```

### View Available Checkpoints
```python
safe_versions = guardian.get_safe_versions()
for v in safe_versions:
    print(f"{v['timestamp']}: {v['stats']['lines']} lines - {v['label']}")
```

### Manual Rollback
```python
paper_path = Path("output/black_hole/paper.tex")
guardian.rollback_to_last_good(paper_path)
```

## 📁 File Locations

### Guardian Files
- `output/black_hole/guardian/` - All checkpoints and manifest
- `output/black_hole/guardian/manifest.json` - Version tracking
- `output/black_hole/guardian/checkpoint_*.tex` - Versioned backups

### Traditional Backups
- `output/black_hole/backups/` - Old-style backups (still available)

## ⚙️ Configuration

### Adjust Protection Thresholds

Edit `utils/content_guardian.py`:

```python
class ContentGuardian:
    # CRITICAL THRESHOLDS
    MIN_ACCEPTABLE_LINES = 2000  # Adjust as needed
    MIN_ACCEPTABLE_SIZE = 140000
    MAX_LINE_LOSS_PERCENT = 5.0
    MAX_SIZE_LOSS_PERCENT = 5.0
```

### Disable Protection (NOT RECOMMENDED)

In your config or workflow:
```python
config.enable_content_protection = False  # DANGEROUS!
```

## 🆘 Troubleshooting

### "Edit blocked" - But I want to apply it anyway

**Option 1:** Fix the issue manually first
```bash
# Review the proposed changes
# Make improvements manually
# Then re-run
```

**Option 2:** Force apply (DANGEROUS)
```python
# Only do this if you're ABSOLUTELY sure
guardian.apply_edit_with_protection(paper_path, new_content, force=True)
```

### "No checkpoints found"

If guardian directory is missing:
```bash
# Use traditional backups
ls output/black_hole/backups/

# Or use emergency recovery
python emergency_recovery.py output/black_hole/paper.tex
```

### "Paper keeps getting corrupted"

**Root cause:** LLM token limit exceeded, response truncated

**Solution:**
1. Switch to model with larger context (Gemini 1.5 Pro = 1M tokens)
2. Split paper into sections
3. Use diff-based revisions only

```bash
# Use high-token model
python main.py ... --model gemini-1.5-pro

# Or Claude
python main.py ... --model claude-3-opus
```

## 📝 Best Practices

1. **Always run pre-flight check** before starting revisions
2. **Monitor the first iteration** to ensure protection is working
3. **Keep backups** in multiple locations (guardian + traditional)
4. **Review blocked edits** - the Guardian might have saved you!
5. **Don't force edits** unless you're absolutely certain
6. **Use high-token models** to avoid truncation (Gemini 1.5 Pro, Claude 3 Opus)

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Pre-flight check | `python preflight_check.py <paper.tex>` |
| Start monitor | `python monitor_content_integrity.py <paper.tex>` |
| Emergency recovery | `python emergency_recovery.py <paper.tex>` |
| View history | `guardian.show_history()` |
| Rollback | `guardian.rollback_to_last_good(paper_path)` |

## 🔐 Protection Guarantees

With all layers active:

- ✅ **Cannot lose >5% of content** in a single edit
- ✅ **Cannot save truncated files** (missing `\end{document}`)
- ✅ **Cannot delete bibliography** or major sections
- ✅ **Auto-recovery** from truncation
- ✅ **Rollback capability** to any checkpoint
- ✅ **Zero data loss** with proper usage

---

**Remember:** The best protection is **prevention**. Always:
1. Run pre-flight check
2. Monitor first iteration
3. Use high-token models
4. Review changes before applying
