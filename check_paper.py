"""
Auto-Protect Papers - Run this before any paper modification
=============================================================
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paper_protection import PaperProtectionSystem

def protect_paper(paper_path: str):
    """Validate and protect a paper."""
    paper_path = Path(paper_path)
    
    print("\n" + "="*70)
    print("🛡️  PAPER PROTECTION CHECK")
    print("="*70)
    print(f"Paper: {paper_path}")
    print("="*70 + "\n")
    
    if not paper_path.exists():
        print(f"❌ Paper not found: {paper_path}")
        return False
    
    # Validate
    is_valid, issues = PaperProtectionSystem.validate_paper(paper_path)
    
    if is_valid:
        print("✅ Paper validation PASSED")
        print(f"   All checks successful!")
        
        # Show stats
        content = paper_path.read_text(encoding='utf-8')
        lines = len(content.split('\n'))
        size = len(content.encode('utf-8'))
        print(f"\n📊 Paper Statistics:")
        print(f"   Lines: {lines:,}")
        print(f"   Size: {size:,} bytes ({size/1024:.1f} KB)")
        
        # Create backup
        backup = PaperProtectionSystem.create_emergency_backup(paper_path)
        print(f"\n💾 Safety backup created:")
        print(f"   {backup}")
        
    else:
        print("❌ Paper validation FAILED\n")
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\n⚠️  DO NOT PROCEED - Fix issues first!")
        
        # Create emergency backup anyway
        backup = PaperProtectionSystem.create_emergency_backup(paper_path)
        print(f"\n💾 Emergency backup created:")
        print(f"   {backup}")
    
    print("\n" + "="*70 + "\n")
    return is_valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_paper.py <paper.tex>")
        sys.exit(1)
    
    paper_path = sys.argv[1]
    is_valid = protect_paper(paper_path)
    
    sys.exit(0 if is_valid else 1)
