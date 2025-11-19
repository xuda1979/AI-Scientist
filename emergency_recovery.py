"""
Emergency recovery script.
Use this if your paper gets corrupted/truncated during AI revisions.
"""
from pathlib import Path
from utils.content_guardian import ContentGuardian
import sys


def emergency_recovery(paper_path: Path):
    """Emergency recovery procedure."""
    print(f"\n{'🚨'*40}")
    print(f"EMERGENCY RECOVERY PROCEDURE")
    print(f"{'🚨'*40}\n")
    
    paper_path = Path(paper_path)
    project_dir = paper_path.parent
    
    print(f"Paper: {paper_path}")
    print(f"Project: {project_dir}\n")
    
    # Check current state
    if paper_path.exists():
        current_size = paper_path.stat().st_size
        current_lines = len(paper_path.read_text(encoding='utf-8', errors='ignore').split('\n'))
        print(f"Current state: {current_lines} lines, {current_size} bytes")
    else:
        print(f"❌ Paper file not found!")
        current_lines = 0
        current_size = 0
    
    # Initialize guardian
    guardian = ContentGuardian(project_dir)
    
    print(f"\n📋 Available Recovery Options:\n")
    
    # Option 1: Last known good
    if guardian.manifest['last_known_good']:
        lkg = guardian.manifest['last_known_good']
        print(f"1. Restore LAST KNOWN GOOD version")
        print(f"   Timestamp: {lkg['timestamp']}")
        print(f"   Lines: {lkg['stats']['lines']:,}")
        print(f"   Size: {lkg['stats']['size']:,} bytes")
        print(f"   Checkpoint: {Path(lkg['checkpoint']).name}")
    
    # Option 2: Recent safe versions
    safe_versions = guardian.get_safe_versions()
    if len(safe_versions) > 1:
        print(f"\n2. Choose from {len(safe_versions)} safe versions:")
        for i, ver in enumerate(safe_versions[-5:], 1):  # Show last 5
            print(f"   [{i}] {ver['timestamp']}: {ver['stats']['lines']} lines, {ver['label']}")
    
    # Option 3: All checkpoints
    all_versions = guardian.manifest['versions']
    print(f"\n3. Browse all {len(all_versions)} checkpoints (including unsafe ones)")
    
    # Option 4: Traditional backups
    backup_dir = project_dir / "backups"
    if backup_dir.exists():
        backups = sorted(backup_dir.glob("*.tex"), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"\n4. Choose from {len(backups)} traditional backups:")
        for backup in backups[:5]:
            size = backup.stat().st_size
            lines = len(backup.read_text(encoding='utf-8', errors='ignore').split('\n'))
            print(f"   {backup.name}: {lines} lines, {size} bytes")
    
    print(f"\n{'='*80}")
    choice = input(f"Select recovery option (1-4, or 'q' to quit): ").strip()
    
    if choice == 'q':
        print("Recovery cancelled.")
        return
    
    if choice == '1':
        # Restore last known good
        if not guardian.manifest['last_known_good']:
            print("❌ No last known good version available!")
            return
        
        print(f"\n🔄 Restoring last known good version...")
        if guardian.rollback_to_last_good(paper_path):
            print(f"\n✅ RECOVERY SUCCESSFUL!")
            verify_recovery(paper_path)
        else:
            print(f"❌ Recovery failed!")
    
    elif choice == '2':
        # Choose from safe versions
        safe_versions = guardian.get_safe_versions()
        if not safe_versions:
            print("❌ No safe versions available!")
            return
        
        print(f"\nSelect a version:")
        for i, ver in enumerate(safe_versions[-10:], 1):
            print(f"  [{i}] {ver['timestamp']}: {ver['stats']['lines']} lines - {ver['label']}")
        
        ver_choice = int(input(f"\nEnter version number (1-{min(10, len(safe_versions))}): "))
        selected = safe_versions[-(11-ver_choice)]
        
        import shutil
        print(f"\n🔄 Restoring version from {selected['timestamp']}...")
        shutil.copy2(selected['checkpoint'], paper_path)
        print(f"\n✅ RECOVERY SUCCESSFUL!")
        verify_recovery(paper_path)
    
    elif choice == '4':
        # Choose from traditional backups
        backups = sorted(backup_dir.glob("*.tex"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        print(f"\nSelect a backup:")
        for i, backup in enumerate(backups[:10], 1):
            size = backup.stat().st_size
            lines = len(backup.read_text(encoding='utf-8', errors='ignore').split('\n'))
            mtime = backup.stat().st_mtime
            from datetime import datetime
            timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  [{i}] {backup.name}")
            print(f"      {timestamp}: {lines} lines, {size} bytes")
        
        backup_choice = int(input(f"\nEnter backup number (1-{min(10, len(backups))}): "))
        selected_backup = backups[backup_choice - 1]
        
        import shutil
        print(f"\n🔄 Restoring backup: {selected_backup.name}...")
        shutil.copy2(selected_backup, paper_path)
        print(f"\n✅ RECOVERY SUCCESSFUL!")
        verify_recovery(paper_path)
    
    else:
        print(f"Invalid choice: {choice}")


def verify_recovery(paper_path: Path):
    """Verify recovery was successful."""
    print(f"\n{'='*80}")
    print(f"VERIFICATION")
    print(f"{'='*80}")
    
    stats = ContentGuardian(paper_path.parent)._get_file_stats(paper_path)
    
    print(f"✓ Lines: {stats['lines']:,}")
    print(f"✓ Size: {stats['size']:,} bytes ({stats['size']/1024:.1f} KB)")
    print(f"✓ Has \\end{{document}}: {'YES' if stats['has_end_document'] else 'NO ❌'}")
    print(f"✓ Has bibliography: {'YES' if stats['has_bibliography'] else 'NO'}")
    print(f"✓ Sections: {stats['section_count']}")
    
    if stats['has_end_document'] and stats['lines'] > 2000:
        print(f"\n✅ Paper appears to be complete and intact!")
        print(f"\n💡 Next steps:")
        print(f"   1. Compile with pdflatex to verify")
        print(f"   2. Review the recovered content")
        print(f"   3. Create a manual backup before proceeding")
    else:
        print(f"\n⚠️  Paper may still have issues. Manual review recommended.")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n🚨 Emergency Recovery Tool")
        print("="*80)
        print("\nUsage: python emergency_recovery.py <paper.tex>")
        print("\nExample:")
        print("  python emergency_recovery.py output/black_hole/paper.tex")
        print("\nThis will:")
        print("  1. Show available recovery points")
        print("  2. Let you choose which version to restore")
        print("  3. Restore the selected version")
        print("  4. Verify the recovery was successful")
        sys.exit(1)
    
    paper_path = Path(sys.argv[1])
    emergency_recovery(paper_path)
