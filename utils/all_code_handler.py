"""
All-Code Mode Handler

This module handles unrestricted code generation, command execution tracking,
and iterative feedback loop for the --all-code flag.

Key Features:
1. Extract and save ANY code files from LLM response (not just simulation.py)
2. Parse and extract execution commands from LLM response
3. Execute commands and capture output
4. Format execution results for next iteration
5. Generate comprehensive diff outputs for all code files
"""

import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


def extract_all_code_blocks(content: str, project_dir: Path) -> Dict[str, str]:
    """
    Extract ALL code blocks with filenames from LLM response.
    
    Supports formats:
    - ```python filename.py
    - ```javascript src/app.js
    - ```bash script.sh
    - File: filename.ext followed by ```
    
    Args:
        content: LLM response text
        project_dir: Project directory path
        
    Returns:
        Dict mapping relative file paths to code content
    """
    code_files = {}
    
    # Pattern 1: ```language filename.ext
    pattern1 = re.compile(
        r'```(\w+)\s+([^\n]+\.[\w]+)\s*\n(.*?)```',
        re.DOTALL | re.MULTILINE
    )
    
    for match in pattern1.finditer(content):
        lang, filepath, code = match.groups()
        filepath = filepath.strip()
        code_files[filepath] = code.strip()
    
    # Pattern 2: File: path/to/file.ext\n```language
    pattern2 = re.compile(
        r'File:\s+([^\n]+\.[\w]+)\s*\n```(?:\w+)?\s*\n(.*?)```',
        re.DOTALL | re.MULTILINE
    )
    
    for match in pattern2.finditer(content):
        filepath, code = match.groups()
        filepath = filepath.strip()
        code_files[filepath] = code.strip()
    
    # Pattern 3: # filename.ext (for Python comments)
    pattern3 = re.compile(
        r'#\s+([^\n]+\.py)\s*\n```python\s*\n(.*?)```',
        re.DOTALL | re.MULTILINE
    )
    
    for match in pattern3.finditer(content):
        filepath, code = match.groups()
        filepath = filepath.strip()
        code_files[filepath] = code.strip()
    
    return code_files


def save_code_files(code_files: Dict[str, str], project_dir: Path, code_subdir: str = "code") -> List[Path]:
    """
    Save extracted code files to project directory.
    
    Args:
        code_files: Dict mapping file paths to content
        project_dir: Project root directory
        code_subdir: Subdirectory for code files (default: "code")
        
    Returns:
        List of saved file paths
    """
    saved_paths = []
    code_dir = project_dir / code_subdir
    code_dir.mkdir(exist_ok=True, parents=True)
    
    for filepath, content in code_files.items():
        # Handle nested paths (e.g., "src/utils/helper.py")
        full_path = code_dir / filepath
        full_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Write file
        full_path.write_text(content, encoding='utf-8')
        saved_paths.append(full_path)
        print(f"  ✓ Saved code file: {filepath}")
    
    return saved_paths


def extract_execution_commands(content: str) -> List[Dict[str, str]]:
    """
    Extract execution commands from LLM response.
    
    Supports formats:
    - Execute: command here
    - Run: command here
    - Command: command here
    - $ command here (shell prompt)
    - > command here (PowerShell prompt)
    - ```bash\ncommand\n```
    
    Returns:
        List of dicts with keys: 'command', 'description', 'shell'
    """
    commands = []
    
    # Pattern 1: Execute:/Run:/Command: directives
    pattern1 = re.compile(
        r'(?:Execute|Run|Command):\s*`?([^`\n]+)`?',
        re.MULTILINE
    )
    
    for match in pattern1.finditer(content):
        cmd = match.group(1).strip()
        commands.append({
            'command': cmd,
            'description': f'Execute: {cmd[:50]}...' if len(cmd) > 50 else f'Execute: {cmd}',
            'shell': 'auto'
        })
    
    # Pattern 2: Shell prompts ($ or >)
    pattern2 = re.compile(
        r'^[$>]\s+(.+)$',
        re.MULTILINE
    )
    
    for match in pattern2.finditer(content):
        cmd = match.group(1).strip()
        if cmd and not cmd.startswith('#'):  # Skip comments
            commands.append({
                'command': cmd,
                'description': f'Shell: {cmd[:50]}...' if len(cmd) > 50 else f'Shell: {cmd}',
                'shell': 'powershell' if content.find('>') != -1 else 'bash'
            })
    
    # Pattern 3: Bash code blocks
    pattern3 = re.compile(
        r'```(?:bash|sh|shell)\s*\n(.*?)```',
        re.DOTALL | re.MULTILINE
    )
    
    for match in pattern3.finditer(content):
        script = match.group(1).strip()
        # Split into individual commands
        for line in script.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                commands.append({
                    'command': line,
                    'description': f'Script: {line[:50]}...' if len(line) > 50 else f'Script: {line}',
                    'shell': 'bash'
                })
    
    return commands


def execute_command(
    cmd: str, 
    cwd: Path, 
    timeout: int = 300,
    shell_type: str = 'auto'
) -> Tuple[int, str, str]:
    """
    Execute a command and capture output.
    
    Args:
        cmd: Command to execute
        cwd: Working directory
        timeout: Timeout in seconds (default: 5 minutes)
        shell_type: 'bash', 'powershell', or 'auto'
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        # Determine shell
        if shell_type == 'auto':
            shell_type = 'powershell' if shutil.which('powershell') else 'bash'
        
        if shell_type == 'powershell':
            full_cmd = ['powershell', '-Command', cmd]
        else:
            full_cmd = cmd
            shell_type = True  # Use shell=True for bash-like commands
        
        # Execute
        result = subprocess.run(
            full_cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=(shell_type == True)
        )
        
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", f"Execution error: {str(e)}"


def execute_commands_and_log(
    commands: List[Dict[str, str]],
    project_dir: Path,
    log_file: str = "execution_log.txt"
) -> Path:
    """
    Execute all commands and log results to a file.
    
    Args:
        commands: List of command dicts
        project_dir: Project directory
        log_file: Log filename
        
    Returns:
        Path to log file
    """
    log_path = project_dir / log_file
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"COMMAND EXECUTION LOG - {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, cmd_info in enumerate(commands, 1):
            cmd = cmd_info['command']
            desc = cmd_info.get('description', cmd)
            shell = cmd_info.get('shell', 'auto')
            
            f.write(f"[{idx}/{len(commands)}] {desc}\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Shell: {shell}\n")
            f.write("-" * 80 + "\n")
            
            # Execute
            exit_code, stdout, stderr = execute_command(cmd, project_dir, shell_type=shell)
            
            # Log results
            f.write(f"Exit Code: {exit_code}\n")
            
            if stdout:
                f.write(f"\nSTDOUT:\n{stdout}\n")
            
            if stderr:
                f.write(f"\nSTDERR:\n{stderr}\n")
            
            if exit_code == 0:
                f.write("\n✓ SUCCESS\n")
            else:
                f.write(f"\n✗ FAILED (exit code: {exit_code})\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"  ✓ Execution log saved: {log_file}")
    return log_path


def generate_code_diffs(
    old_files: Dict[str, str],
    new_files: Dict[str, str],
    project_dir: Path,
    iteration: int
) -> Path:
    """
    Generate unified diffs for all code files.
    
    Args:
        old_files: Dict of old file paths -> content
        new_files: Dict of new file paths -> content
        project_dir: Project directory
        iteration: Iteration number
        
    Returns:
        Path to diff file
    """
    from difflib import unified_diff
    
    diff_dir = project_dir / "diffs"
    diff_dir.mkdir(exist_ok=True)
    
    diff_file = diff_dir / f"code_changes_iteration_{iteration}.diff"
    
    with open(diff_file, 'w', encoding='utf-8') as f:
        f.write(f"Code Changes - Iteration {iteration}\n")
        f.write("=" * 80 + "\n\n")
        
        # All unique file paths
        all_paths = set(old_files.keys()) | set(new_files.keys())
        
        for filepath in sorted(all_paths):
            old_content = old_files.get(filepath, "")
            new_content = new_files.get(filepath, "")
            
            if old_content == new_content:
                continue  # No changes
            
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            
            diff = unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{filepath}",
                tofile=f"b/{filepath}",
                lineterm=""
            )
            
            diff_text = ''.join(diff)
            if diff_text:
                f.write(f"--- {filepath} ---\n")
                f.write(diff_text)
                f.write("\n\n")
    
    print(f"  ✓ Code diffs saved: {diff_file.name}")
    return diff_file


def format_execution_results_for_llm(
    log_path: Path,
    diff_path: Optional[Path] = None
) -> str:
    """
    Format execution results and diffs for inclusion in next LLM iteration.
    
    Args:
        log_path: Path to execution log
        diff_path: Optional path to code diff file
        
    Returns:
        Formatted string for LLM context
    """
    content = []
    
    content.append("=" * 80)
    content.append("PREVIOUS ITERATION EXECUTION RESULTS")
    content.append("=" * 80)
    content.append("")
    
    # Add execution log
    if log_path and log_path.exists():
        log_content = log_path.read_text(encoding='utf-8')
        content.append("## Command Execution Log")
        content.append("")
        content.append(log_content)
        content.append("")
    
    # Add code diffs
    if diff_path and diff_path.exists():
        diff_content = diff_path.read_text(encoding='utf-8')
        content.append("## Code Changes (Git Diff Format)")
        content.append("")
        content.append(diff_content)
        content.append("")
    
    content.append("=" * 80)
    content.append("Please review the execution results above and:")
    content.append("1. Fix any errors or failures")
    content.append("2. Improve code based on actual output")
    content.append("3. Add new features or refinements")
    content.append("4. Provide new execution commands if needed")
    content.append("=" * 80)
    
    return "\n".join(content)


def create_all_code_prompt_supplement(iteration: int, has_previous_results: bool = False) -> str:
    """
    Create prompt supplement for all-code mode.
    
    Args:
        iteration: Current iteration number
        has_previous_results: Whether previous execution results exist
        
    Returns:
        Prompt text to add to LLM request
    """
    prompt = []
    
    prompt.append("\n" + "=" * 80)
    prompt.append("ALL-CODE MODE ENABLED")
    prompt.append("=" * 80)
    prompt.append("")
    prompt.append("You can generate ANY code files (not restricted to simulation.py):")
    prompt.append("")
    prompt.append("**Code File Format:**")
    prompt.append("```python path/to/file.py")
    prompt.append("# Your code here")
    prompt.append("```")
    prompt.append("")
    prompt.append("Or use:")
    prompt.append("File: path/to/file.py")
    prompt.append("```python")
    prompt.append("# Your code here")
    prompt.append("```")
    prompt.append("")
    prompt.append("**Supported Languages:** Python, JavaScript, TypeScript, C++, Java, Go, Rust, R, SQL, etc.")
    prompt.append("")
    prompt.append("**Execution Commands:**")
    prompt.append("Specify commands to run using:")
    prompt.append("- Execute: python main.py")
    prompt.append("- Run: npm test")
    prompt.append("- Command: cargo build --release")
    prompt.append("")
    prompt.append("Or use shell format:")
    prompt.append("```bash")
    prompt.append("python train.py --epochs 10")
    prompt.append("python evaluate.py --model best_model.pkl")
    prompt.append("```")
    prompt.append("")
    
    if has_previous_results:
        prompt.append("**Previous execution results are shown above.**")
        prompt.append("Please review errors, fix issues, and iterate on the codebase.")
    else:
        prompt.append("**This is iteration #{}.** Generate code and commands.".format(iteration))
    
    prompt.append("")
    prompt.append("**Workflow:**")
    prompt.append("1. Generate/update code files")
    prompt.append("2. Provide execution commands")
    prompt.append("3. Review results in next iteration")
    prompt.append("4. Iterate until working correctly")
    prompt.append("")
    prompt.append("=" * 80)
    
    return "\n".join(prompt)
