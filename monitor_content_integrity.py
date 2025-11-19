"""
Emergency content integrity monitor.
Runs in background and alerts on suspicious file changes.
"""
import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.content_guardian import ContentGuardian


class PaperIntegrityMonitor(FileSystemEventHandler):
    """Monitor paper.tex for suspicious changes."""
    
    def __init__(self, paper_path: Path):
        self.paper_path = paper_path
        self.project_dir = paper_path.parent
        self.guardian = ContentGuardian(self.project_dir)
        self.last_stats = self.guardian._get_file_stats(paper_path)
        
        print(f"\n{'='*80}")
        print(f"📡 CONTENT INTEGRITY MONITOR STARTED")
        print(f"{'='*80}")
        print(f"Monitoring: {paper_path}")
        print(f"Baseline: {self.last_stats['lines']} lines, {self.last_stats['size']} bytes")
        print(f"{'='*80}\n")
    
    def on_modified(self, event):
        if event.src_path != str(self.paper_path):
            return
        
        # Wait a bit for write to complete
        time.sleep(0.5)
        
        new_stats = self.guardian._get_file_stats(self.paper_path)
        
        # Calculate change
        line_change = new_stats['lines'] - self.last_stats['lines']
        size_change = new_stats['size'] - self.last_stats['size']
        line_pct = (line_change / self.last_stats['lines']) * 100 if self.last_stats['lines'] > 0 else 0
        
        # Alert on suspicious changes
        if line_change < -100:  # Lost more than 100 lines
            print(f"\n{'🚨'*40}")
            print(f"⚠️  ALERT: LARGE CONTENT LOSS DETECTED!")
            print(f"{'🚨'*40}")
            print(f"Lost {abs(line_change)} lines ({line_pct:.1f}%)")
            print(f"Before: {self.last_stats['lines']} lines, {self.last_stats['size']} bytes")
            print(f"After:  {new_stats['lines']} lines, {new_stats['size']} bytes")
            
            # Check for truncation
            if not new_stats['has_end_document']:
                print(f"\n🚨 CRITICAL: File appears TRUNCATED (missing \\end{{document}})!")
                print(f"Attempting automatic recovery...")
                
                # Attempt rollback
                if self.guardian.rollback_to_last_good(self.paper_path):
                    print(f"✓ Automatic rollback successful!")
                    recovered_stats = self.guardian._get_file_stats(self.paper_path)
                    print(f"Recovered: {recovered_stats['lines']} lines, {recovered_stats['size']} bytes")
                else:
                    print(f"❌ Automatic rollback failed - manual intervention required!")
            
            print(f"{'🚨'*40}\n")
        
        elif line_change > 50 or line_change < -20:
            print(f"\n📝 Paper modified: {line_change:+d} lines ({line_pct:+.1f}%)")
        
        self.last_stats = new_stats


def start_monitor(paper_path: Path):
    """Start monitoring a paper file."""
    paper_path = Path(paper_path).resolve()
    
    if not paper_path.exists():
        print(f"Error: Paper not found: {paper_path}")
        return
    
    event_handler = PaperIntegrityMonitor(paper_path)
    observer = Observer()
    observer.schedule(event_handler, str(paper_path.parent), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print(f"\n\n{'='*80}")
        print(f"📡 MONITOR STOPPED")
        print(f"{'='*80}\n")
    
    observer.join()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_content_integrity.py <paper.tex>")
        print("\nExample:")
        print("  python monitor_content_integrity.py output/black_hole/paper.tex")
        sys.exit(1)
    
    paper_path = Path(sys.argv[1])
    start_monitor(paper_path)
