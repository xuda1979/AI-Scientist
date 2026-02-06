#!/usr/bin/env python3
"""
AI Agent for the AI Scientist IDE.
This module handles AI-powered code assistance and file modifications.
"""
import json
import sys
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ai.chat import chat
except ImportError:
    chat = None


def extract_file_changes(response: str) -> List[Dict[str, Any]]:
    """Extract file modification instructions from AI response."""
    changes = []
    
    # Look for JSON blocks with file changes
    json_pattern = r'```json\s*(\{[^`]+\})\s*```'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match)
            if 'action' in data and 'path' in data:
                changes.append(data)
        except json.JSONDecodeError:
            continue
    
    # Also look for code blocks with file paths
    code_pattern = r'```(\w+)?\s*\n?#\s*(?:File|Path):\s*([^\n]+)\n([\s\S]*?)```'
    code_matches = re.findall(code_pattern, response, re.IGNORECASE)
    
    for lang, filepath, content in code_matches:
        filepath = filepath.strip()
        if filepath:
            changes.append({
                'action': 'modify',
                'path': filepath,
                'content': content.strip(),
            })
    
    return changes


def build_system_prompt(context: Dict[str, Any]) -> str:
    """Build system prompt with workspace context."""
    prompt = """You are an AI coding assistant (Copilot) integrated into the AI Scientist IDE.
You help users with coding tasks including:
- Writing and modifying code
- Refactoring and optimization
- Bug fixing and debugging
- Documentation generation
- Code explanation

When you need to create or modify files, use this JSON format in your response:
```json
{"action": "create", "path": "/full/path/to/file.py", "content": "file content here"}
```

Or for modifications:
```json
{"action": "modify", "path": "/full/path/to/file.py", "content": "new file content"}
```

For deletions:
```json
{"action": "delete", "path": "/full/path/to/file.py"}
```

Always provide clear explanations along with any code changes.
"""
    
    workspace_path = context.get('workspacePath')
    if workspace_path:
        prompt += f"\n\nCurrent workspace: {workspace_path}"
    
    open_files = context.get('openFiles', [])
    if open_files:
        prompt += f"\n\nOpen files: {', '.join(f.get('path', '') for f in open_files)}"
    
    active_file = context.get('activeFile')
    if active_file:
        prompt += f"\n\n--- Currently editing: {active_file.get('path', '')} ---\n"
        content = active_file.get('content', '')
        # Truncate very long files
        if len(content) > 10000:
            content = content[:10000] + "\n... (truncated)"
        prompt += f"```\n{content}\n```"
    
    return prompt


def process_agent_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process an agent chat request."""
    model = input_data.get('model', 'gpt-4o')
    messages = input_data.get('messages', [])
    context = input_data.get('context', {})
    
    if not messages:
        return {'error': 'No messages provided'}
    
    # Build full message list with system prompt
    system_prompt = build_system_prompt(context)
    full_messages = [{'role': 'system', 'content': system_prompt}] + messages
    
    try:
        if chat is None:
            # Fallback response if chat module not available
            return {
                'response': "AI chat module not available. Please check your configuration.",
                'fileChanges': [],
            }
        
        # Call the AI chat function
        response, tokens = chat(
            messages=full_messages,
            model=model,
            temperature=0.7,
            request_timeout=120,
            prompt_type='agent',
        )
        
        # Extract any file changes from the response
        file_changes = extract_file_changes(response)
        
        return {
            'response': response,
            'fileChanges': file_changes,
            'tokensUsed': tokens,
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'response': f"Error: {str(e)}",
            'fileChanges': [],
        }


def main():
    """Main entry point for the agent script."""
    try:
        # Read input from stdin
        input_text = sys.stdin.read()
        input_data = json.loads(input_text)
        
        # Process the request
        result = process_agent_request(input_data)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'Invalid JSON input: {str(e)}'}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
