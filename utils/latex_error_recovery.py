"""
LaTeX Error Recovery Module
Automatically fixes common LaTeX compilation errors by analyzing log files.
"""
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LaTeXErrorRecovery:
    """Handles automatic recovery from common LaTeX compilation errors."""
    
    def __init__(self, tex_file: Path, log_file: Optional[Path] = None):
        """
        Initialize error recovery for a LaTeX file.
        
        Args:
            tex_file: Path to the .tex file
            log_file: Path to the .log file (if None, will use tex_file.with_suffix('.log'))
        """
        self.tex_file = tex_file
        self.log_file = log_file or tex_file.with_suffix('.log')
        self.errors: List[Dict] = []
        self.fixes_applied: List[str] = []
    
    def analyze_log(self) -> List[Dict]:
        """
        Parse LaTeX log file to extract error information.
        
        Returns:
            List of error dictionaries with keys: type, line, message, context
        """
        if not self.log_file.exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return []
        
        log_content = self.log_file.read_text(encoding='utf-8', errors='ignore')
        errors = []
        
        # Pattern for standard LaTeX errors
        error_pattern = re.compile(
            r'! (.+?)\n'  # Error message
            r'l\.(\d+)\s+(.+?)(?:\n|$)',  # Line number and context
            re.DOTALL
        )
        
        for match in error_pattern.finditer(log_content):
            error_msg = match.group(1).strip()
            line_num = int(match.group(2))
            context = match.group(3).strip()
            
            errors.append({
                'type': self._classify_error(error_msg),
                'line': line_num,
                'message': error_msg,
                'context': context
            })
        
        # Check for emergency stop (truncated file)
        if 'Emergency stop' in log_content:
            errors.append({
                'type': 'truncated_file',
                'line': None,
                'message': 'Emergency stop - file may be truncated',
                'context': ''
            })
        
        # Check for missing \end{document}
        if 'no legal \\end found' in log_content or '\\end occurred when' in log_content:
            errors.append({
                'type': 'missing_end_document',
                'line': None,
                'message': 'Missing \\end{document}',
                'context': ''
            })
        
        # Check for pgfplots errors
        if 'pgfplots' in log_content.lower() and 'unknown function' in log_content.lower():
            # Extract the problematic function
            func_match = re.search(r"Unknown function '([^']+)'", log_content)
            if func_match:
                errors.append({
                    'type': 'pgfplots_parsing',
                    'line': None,
                    'message': f"pgfplots parsing error: Unknown function '{func_match.group(1)}'",
                    'context': func_match.group(1)
                })
        
        # Check for pgfplotstable errors
        if 'pgfplotstable' in log_content and ('Incomplete \\ifx' in log_content or 'Undefined control sequence' in log_content):
            errors.append({
                'type': 'pgfplotstable_error',
                'line': None,
                'message': 'pgfplotstable syntax error - likely misplaced command',
                'context': ''
            })
        
        self.errors = errors
        return errors
    
    def _classify_error(self, error_msg: str) -> str:
        """Classify error type based on message."""
        error_msg_lower = error_msg.lower()
        
        if 'undefined control sequence' in error_msg_lower:
            return 'undefined_command'
        elif 'missing' in error_msg_lower:
            if '\\begin' in error_msg_lower or '\\end' in error_msg_lower:
                return 'unbalanced_environment'
            else:
                return 'missing_character'
        elif 'emergency stop' in error_msg_lower:
            return 'emergency_stop'
        elif 'dimension too large' in error_msg_lower:
            return 'dimension_overflow'
        elif 'file not found' in error_msg_lower:
            return 'missing_file'
        else:
            return 'unknown'
    
    def auto_fix(self) -> Tuple[bool, str]:
        """
        Attempt to automatically fix errors found in the LaTeX file.
        
        Returns:
            Tuple of (success, fixed_content)
        """
        if not self.tex_file.exists():
            logger.error(f"TeX file not found: {self.tex_file}")
            return False, ""
        
        content = self.tex_file.read_text(encoding='utf-8', errors='ignore')
        original_content = content
        
        # Analyze errors if not already done
        if not self.errors:
            self.analyze_log()
        
        # Apply fixes based on error types
        for error in self.errors:
            error_type = error['type']
            
            if error_type == 'missing_end_document' or error_type == 'truncated_file':
                content = self._fix_missing_end_document(content)
            
            elif error_type == 'pgfplots_parsing':
                content = self._fix_pgfplots_parsing(content, error.get('context', ''))
            
            elif error_type == 'pgfplotstable_error':
                content = self._fix_pgfplotstable_errors(content)
            
            elif error_type == 'unbalanced_environment':
                content = self._fix_unbalanced_environments(content)
        
        # Generic cleanup
        content = self._generic_cleanup(content)
        
        success = content != original_content
        return success, content
    
    def _fix_missing_end_document(self, content: str) -> str:
        """Add missing \\end{document} if not present."""
        if not re.search(r'\\end\{document\}', content):
            # Find last complete structure
            last_para = content.rfind('\n\n')
            last_brace = content.rfind('}')
            last_period = content.rfind('.')
            
            # Cut at a reasonable point
            cutoff = max(last_para, last_brace, last_period)
            if cutoff > len(content) - 500:  # Within last 500 chars
                content = content[:cutoff+1] + '\n\n\\end{document}\n'
                self.fixes_applied.append("Added missing \\end{document}")
                logger.info("Fix applied: Added \\end{document}")
            else:
                # Just append
                content += '\n\n\\end{document}\n'
                self.fixes_applied.append("Appended \\end{document}")
                logger.info("Fix applied: Appended \\end{document}")
        
        return content
    
    def _fix_pgfplots_parsing(self, content: str, problematic_func: str) -> str:
        """Fix pgfplots parsing errors from parentheses in symbolic coordinates."""
        
        # Find axis blocks with symbolic coords
        def fix_symbolic_coords(match):
            axis_block = match.group(0)
            modified = False
            
            # Check for symbolic y coords with parentheses
            symbolic_match = re.search(
                r'symbolic\s+y\s+coords\s*=\s*\{([^}]+)\}',
                axis_block
            )
            if symbolic_match and ('(' in symbolic_match.group(1) or ')' in symbolic_match.group(1)):
                coords = symbolic_match.group(1)
                coord_list = [c.strip() for c in coords.split(',')]
                
                # Remove parentheses from coordinates
                clean_coords = [re.sub(r'[()]', '', c).replace('  ', ' ') for c in coord_list]
                
                # Replace with numeric approach
                ytick_values = ','.join(str(i) for i in range(len(clean_coords)))
                yticklabels = ','.join(clean_coords)
                
                # Remove symbolic y coords
                axis_block = re.sub(
                    r'symbolic\s+y\s+coords\s*=\s*\{[^}]+\}[,\s]*',
                    '',
                    axis_block
                )
                
                # Replace ytick=data
                axis_block = re.sub(
                    r'ytick\s*=\s*data',
                    f'ytick={{{ytick_values}}},\n        yticklabels={{{yticklabels}}}',
                    axis_block
                )
                
                modified = True
                logger.info("Fix applied: Converted symbolic y coords to numeric")
            
            # Same for x coords
            symbolic_match_x = re.search(
                r'symbolic\s+x\s+coords\s*=\s*\{([^}]+)\}',
                axis_block
            )
            if symbolic_match_x and ('(' in symbolic_match_x.group(1) or ')' in symbolic_match_x.group(1)):
                coords = symbolic_match_x.group(1)
                coord_list = [c.strip() for c in coords.split(',')]
                clean_coords = [re.sub(r'[()]', '', c).replace('  ', ' ') for c in coord_list]
                
                xtick_values = ','.join(str(i) for i in range(len(clean_coords)))
                xticklabels = ','.join(clean_coords)
                
                axis_block = re.sub(
                    r'symbolic\s+x\s+coords\s*=\s*\{[^}]+\}[,\s]*',
                    '',
                    axis_block
                )
                axis_block = re.sub(
                    r'xtick\s*=\s*data',
                    f'xtick={{{xtick_values}}},\n        xticklabels={{{xticklabels}}}',
                    axis_block
                )
                
                modified = True
            
            if modified:
                self.fixes_applied.append("Fixed pgfplots symbolic coordinates")
            
            return axis_block
        
        content = re.sub(
            r'\\begin\{axis\}.*?\\end\{axis\}',
            fix_symbolic_coords,
            content,
            flags=re.DOTALL
        )
        
        return content
    
    def _fix_pgfplotstable_errors(self, content: str) -> str:
        """Fix pgfplotstable errors by removing problematic table commands."""
        
        # Find tables with pgfplotstable commands inside
        def fix_table(match):
            table_content = match.group(0)
            
            if '\\pgfplotstable' in table_content:
                # Extract the data file reference
                data_match = re.search(r'\\pgfplotstableread.*?\{([^}]+)\}', table_content)
                if data_match:
                    data_file = data_match.group(1)
                    logger.warning(f"Removing pgfplotstable commands for {data_file} - convert to manual table")
                
                # Remove pgfplotstable commands but keep the table structure
                # This is a basic fix - the table data needs to be manually inserted
                table_content = re.sub(r'\\pgfplotstableread[^}]+\}[^}]+\}', '', table_content)
                table_content = re.sub(r'\\pgfplotstabletypeset\[.*?\]\{[^}]+\}', 
                                      '% Manual table data needed here', table_content)
                
                self.fixes_applied.append("Removed problematic pgfplotstable commands")
                logger.info("Fix applied: Removed pgfplotstable from tabular environment")
            
            return table_content
        
        content = re.sub(
            r'\\begin\{table\}.*?\\end\{table\}',
            fix_table,
            content,
            flags=re.DOTALL
        )
        
        return content
    
    def _fix_unbalanced_environments(self, content: str) -> str:
        """Attempt to balance unbalanced environments."""
        
        # Find all \begin and \end commands
        begins = [(m.start(), m.group(1)) for m in re.finditer(r'\\begin\{(\w+)\}', content)]
        ends = [(m.start(), m.group(1)) for m in re.finditer(r'\\end\{(\w+)\}', content)]
        
        # Track which environments are unbalanced
        from collections import Counter
        begin_counts = Counter([env for _, env in begins])
        end_counts = Counter([env for _, env in ends])
        
        for env in begin_counts:
            if begin_counts[env] > end_counts.get(env, 0):
                # More begins than ends - add missing ends
                missing = begin_counts[env] - end_counts.get(env, 0)
                logger.info(f"Adding {missing} missing \\end{{{env}}} command(s)")
                
                # Add before \end{document}
                end_doc_pos = content.rfind('\\end{document}')
                if end_doc_pos > 0:
                    missing_ends = '\n'.join([f'\\end{{{env}}}'] * missing)
                    content = content[:end_doc_pos] + missing_ends + '\n\n' + content[end_doc_pos:]
                    self.fixes_applied.append(f"Added {missing} missing \\end{{{env}}}")
        
        return content
    
    def _generic_cleanup(self, content: str) -> str:
        """Apply generic cleanup fixes."""
        
        # Remove duplicate \end{document}
        end_doc_count = content.count('\\end{document}')
        if end_doc_count > 1:
            # Keep only the last one
            parts = content.split('\\end{document}')
            content = ''.join(parts[:-1]) + '\\end{document}' + parts[-1]
            self.fixes_applied.append(f"Removed {end_doc_count - 1} duplicate \\end{{document}} commands")
        
        # Ensure single newline before \end{document}
        content = re.sub(r'\n{3,}\\end\{document\}', r'\n\n\\end{document}', content)
        
        return content
    
    def get_error_summary(self) -> str:
        """Get a human-readable summary of errors found."""
        if not self.errors:
            return "No errors found."
        
        summary = f"Found {len(self.errors)} error(s):\n"
        for i, error in enumerate(self.errors, 1):
            line_info = f" (line {error['line']})" if error['line'] else ""
            summary += f"{i}. {error['type']}{line_info}: {error['message']}\n"
        
        return summary


def recover_from_compilation_failure(tex_file: Path, max_attempts: int = 3) -> Tuple[bool, List[str]]:
    """
    Attempt to recover from LaTeX compilation failure.
    
    Args:
        tex_file: Path to the .tex file
        max_attempts: Maximum number of recovery attempts
        
    Returns:
        Tuple of (success, list_of_fixes_applied)
    """
    logger.info(f"Attempting error recovery for {tex_file}")
    
    recovery = LaTeXErrorRecovery(tex_file)
    all_fixes = []
    
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Recovery attempt {attempt}/{max_attempts}")
        
        # Analyze errors
        errors = recovery.analyze_log()
        if not errors:
            logger.info("No errors found in log file")
            break
        
        logger.info(f"Found {len(errors)} errors")
        logger.debug(recovery.get_error_summary())
        
        # Attempt auto-fix
        success, fixed_content = recovery.auto_fix()
        
        if success:
            # Save the fixed content
            backup = tex_file.with_suffix('.tex.recovery_backup')
            tex_file.rename(backup)
            tex_file.write_text(fixed_content, encoding='utf-8')
            
            all_fixes.extend(recovery.fixes_applied)
            logger.info(f"Applied {len(recovery.fixes_applied)} fixes, saved backup to {backup.name}")
            
            # Try recompiling (caller should do this)
            return True, all_fixes
        else:
            logger.warning(f"Auto-fix attempt {attempt} did not produce changes")
    
    if all_fixes:
        return True, all_fixes
    else:
        return False, []


if __name__ == "__main__":
    # Test the recovery system
    import sys
    
    if len(sys.argv) > 1:
        tex_path = Path(sys.argv[1])
        
        recovery = LaTeXErrorRecovery(tex_path)
        errors = recovery.analyze_log()
        
        print(f"\n=== Error Analysis for {tex_path.name} ===")
        print(recovery.get_error_summary())
        
        if errors:
            print("\n=== Attempting Auto-Fix ===")
            success, fixed_content = recovery.auto_fix()
            
            if success:
                print(f"Fixes applied: {len(recovery.fixes_applied)}")
                for fix in recovery.fixes_applied:
                    print(f"  - {fix}")
                
                save = input("\nSave fixed version? (y/n): ")
                if save.lower() == 'y':
                    backup = tex_path.with_suffix('.tex.bak')
                    tex_path.rename(backup)
                    tex_path.write_text(fixed_content, encoding='utf-8')
                    print(f"Saved! Backup: {backup}")
            else:
                print("No fixes could be applied automatically.")
    else:
        print("Usage: python latex_error_recovery.py <path_to_tex_file>")
