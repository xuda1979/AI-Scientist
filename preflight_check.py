"""
Pre-flight safety check before running AI revisions.
Use this to verify your paper is in good condition before starting automated revisions.
"""
from pathlib import Path
from utils.content_guardian import ContentGuardian
import sys


def preflight_check(paper_path: Path) -> bool:
    """
    Run comprehensive preflight checks.
    
    Returns:
        True if safe to proceed, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🛫 PRE-FLIGHT SAFETY CHECK")
    print(f"{'='*80}\n")
    
    paper_path = Path(paper_path)
    
    if not paper_path.exists():
        print(f"❌ ABORT: Paper file not found: {paper_path}")
        return False
    
    project_dir = paper_path.parent
    guardian = ContentGuardian(project_dir)
    
    # Create initial checkpoint
    print(f"Creating initial safety checkpoint...")
    checkpoint = guardian.create_checkpoint(paper_path, "preflight")
    print(f"✓ Checkpoint: {Path(checkpoint).name}\n")
    
    # Get file stats
    stats = guardian._get_file_stats(paper_path)
    
    print(f"📄 Paper Statistics:")
    print(f"   Lines: {stats['lines']:,}")
    print(f"   Size: {stats['size']:,} bytes ({stats['size']/1024:.1f} KB)")
    print(f"   Sections: {stats['section_count']}")
    print(f"   Subsections: {stats['subsection_count']}")
    print(f"\n✓ Structural Checks:")
    print(f"   Has \\end{{document}}: {'✓' if stats['has_end_document'] else '❌'}")
    print(f"   Has bibliography: {'✓' if stats['has_bibliography'] else '❌'}")
    
    # Safety assessment
    issues = []
    
    if stats['lines'] < guardian.MIN_ACCEPTABLE_LINES:
        issues.append(f"⚠️  Paper is too short: {stats['lines']} lines (minimum: {guardian.MIN_ACCEPTABLE_LINES})")
    
    if stats['size'] < guardian.MIN_ACCEPTABLE_SIZE:
        issues.append(f"⚠️  File size too small: {stats['size']} bytes (minimum: {guardian.MIN_ACCEPTABLE_SIZE})")
    
    if not stats['has_end_document']:
        issues.append(f"❌ CRITICAL: Missing \\end{{document}} - file appears truncated!")
    
    if not stats['has_bibliography']:
        issues.append(f"⚠️  No bibliography found - may be incomplete")
    
    if stats['section_count'] < 5:
        issues.append(f"⚠️  Very few sections ({stats['section_count']}) - may be incomplete")
    
    print(f"\n{'='*80}")
    if issues:
        print(f"🚨 ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        print(f"\n⛔ RECOMMENDATION: Do NOT proceed with automated revisions!")
        print(f"   Fix these issues manually first.")
        print(f"{'='*80}\n")
        return False
    else:
        print(f"✅ ALL CHECKS PASSED - Safe to proceed!")
        print(f"\n💡 Recommendations:")
        print(f"   1. Keep this terminal open to monitor progress")
        print(f"   2. Guardian will create checkpoints before each change")
        print(f"   3. Any destructive edits will be automatically blocked")
        print(f"   4. You can rollback to this checkpoint anytime")
        print(f"\n🛡️  Protection layers active:")
        print(f"   ✓ Content Guardian with automatic rollback")
        print(f"   ✓ Size/line count validation")
        print(f"   ✓ Structural integrity checks")
        print(f"   ✓ Emergency checkpoints")
        print(f"{'='*80}\n")
        return True


def post_flight_check(paper_path: Path):
    """Run post-flight check after revisions complete."""
    print(f"\n{'='*80}")
    print(f"🛬 POST-FLIGHT SAFETY CHECK")
    print(f"{'='*80}\n")
    
    paper_path = Path(paper_path)
    project_dir = paper_path.parent
    guardian = ContentGuardian(project_dir)
    
    # Show history
    guardian.show_history()
    
    # Verify final state
    final_stats = guardian._get_file_stats(paper_path)
    
    print(f"📄 Final Paper State:")
    print(f"   Lines: {final_stats['lines']:,}")
    print(f"   Size: {final_stats['size']:,} bytes")
    print(f"   Complete: {'✓' if final_stats['has_end_document'] else '❌'}")
    
    # Compare to last known good
    if guardian.manifest['last_known_good']:
        lkg = guardian.manifest['last_known_good']
        line_diff = final_stats['lines'] - lkg['stats']['lines']
        size_diff = final_stats['size'] - lkg['stats']['size']
        
        print(f"\n📊 Changes since last known good:")
        print(f"   Lines: {line_diff:+d}")
        print(f"   Size: {size_diff:+d} bytes")
        
        if abs(line_diff) > 100:
            print(f"\n⚠️  Significant change in content!")
            print(f"   Review the changes carefully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n🛫 Pre-Flight Safety Check")
        print("="*80)
        print("\nUsage: python preflight_check.py <paper.tex>")
        print("\nExample:")
        print("  python preflight_check.py output/black_hole/paper.tex")
        print("\nThis will:")
        print("  1. Verify paper integrity")
        print("  2. Create safety checkpoint")
        print("  3. Report any issues")
        print("  4. Recommend whether to proceed with automated revisions")
        sys.exit(1)
    
    paper_path = Path(sys.argv[1])
    safe = preflight_check(paper_path)
    
    sys.exit(0 if safe else 1)
