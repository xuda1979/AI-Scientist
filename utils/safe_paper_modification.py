"""
Safe Paper Modification - High-Level Integration
================================================
Safe, validated paper modifications with unlimited output and protection.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.safe_llm_wrapper import safe_llm_call, create_messages
from utils.document_protection import DocumentValidator, DocumentChunker, SafeFileWriter
from config.gpt_unlimited_config import get_model_config, estimate_tokens


class SafePaperModifier:
    """High-level interface for safe paper modifications."""
    
    def __init__(
        self,
        model: str = "gpt-5-pro",
        verbose: bool = True,
        enable_chunking: bool = True,
        validate_output: bool = True,
        create_backups: bool = True
    ):
        """
        Initialize safe paper modifier.
        
        Args:
            model: LLM model to use
            verbose: Whether to print progress
            enable_chunking: Whether to enable automatic chunking for large documents
            validate_output: Whether to validate output before saving
            create_backups: Whether to create backups before modifications
        """
        self.model = model
        self.verbose = verbose
        self.enable_chunking = enable_chunking
        self.validate_output = validate_output
        self.create_backups = create_backups
        self.config = get_model_config(model)
    
    def modify_paper(
        self,
        paper_path: Path,
        modification_instructions: str,
        system_prompt: Optional[str] = None,
        **llm_kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Safely modify a paper with validation and protection.
        
        Args:
            paper_path: Path to paper file
            modification_instructions: Instructions for modifications
            system_prompt: Optional system prompt (default provided)
            **llm_kwargs: Additional arguments for LLM call
            
        Returns:
            (success, error_message)
        """
        paper_path = Path(paper_path)
        
        if not paper_path.exists():
            return False, f"Paper file does not exist: {paper_path}"
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📄 Safe Paper Modification")
            print(f"{'='*70}")
            print(f"File: {paper_path}")
            print(f"Model: {self.model}")
            print(f"Unlimited output: ✅")
            print(f"{'='*70}\n")
        
        # Read original content
        try:
            original_content = paper_path.read_text(encoding='utf-8')
            original_lines = len(original_content.split('\n'))
            original_size = len(original_content)
            
            if self.verbose:
                print(f"📖 Original paper:")
                print(f"   Lines: {original_lines:,}")
                print(f"   Size: {original_size:,} bytes")
                print(f"   Estimated tokens: ~{estimate_tokens(original_content):,}")
        except Exception as e:
            return False, f"Failed to read paper: {e}"
        
        # Validate original
        if self.validate_output:
            is_valid, issues = DocumentValidator.validate_latex(original_content)
            if not is_valid:
                print(f"\n⚠️  Warning: Original paper has validation issues:")
                for issue in issues:
                    print(f"    - {issue}")
                print(f"   Continuing anyway...\n")
        
        # Determine if chunking is needed
        chunk_threshold = self.config.get("chunking", {}).get("chunk_size", 500)
        needs_chunking = self.enable_chunking and original_lines > chunk_threshold
        
        if needs_chunking:
            if self.verbose:
                print(f"\n📦 Document is large ({original_lines} lines > {chunk_threshold})")
                print(f"   Chunking enabled for safety")
            
            return self._modify_with_chunking(
                paper_path,
                original_content,
                modification_instructions,
                system_prompt,
                **llm_kwargs
            )
        else:
            return self._modify_direct(
                paper_path,
                original_content,
                modification_instructions,
                system_prompt,
                **llm_kwargs
            )
    
    def _modify_direct(
        self,
        paper_path: Path,
        original_content: str,
        modification_instructions: str,
        system_prompt: Optional[str],
        **llm_kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Direct modification without chunking."""
        
        if system_prompt is None:
            system_prompt = r"""You are an expert LaTeX editor and academic writer.
Your task is to modify research papers according to provided instructions.

CRITICAL REQUIREMENTS:
1. Output the COMPLETE modified paper - never truncate or abbreviate
2. Preserve ALL sections, equations, citations, and references
3. Maintain proper LaTeX structure and formatting
4. Keep ALL \begin{} and \end{} environments properly paired
5. Do not use placeholders like "...rest of content..." or "[content continues]"
6. Output MUST be valid, compilable LaTeX
7. If paper is long, output ALL content - no shortcuts

Remember: The user needs the FULL paper, not a summary or partial output."""
        
        # Create user prompt
        user_prompt = f"""Please modify the following LaTeX paper according to these instructions:

INSTRUCTIONS:
{modification_instructions}

PAPER CONTENT:
{original_content}

OUTPUT THE COMPLETE MODIFIED PAPER BELOW (full content, no truncation):
"""
        
        messages = create_messages(system_prompt, user_prompt)
        
        # Call LLM with unlimited output
        try:
            if self.verbose:
                print(f"\n🤖 Calling {self.model} with unlimited output...")
            
            modified_content = safe_llm_call(
                messages=messages,
                model=self.model,
                max_tokens=None,  # UNLIMITED!
                verbose=self.verbose,
                **llm_kwargs
            )
            
            # Validate modified content
            if self.validate_output:
                if self.verbose:
                    print(f"\n🔍 Validating modified content...")
                
                is_valid, issues = DocumentValidator.validate_latex(
                    modified_content,
                    original_content
                )
                
                if not is_valid:
                    error_msg = "Modified content validation failed:\n"
                    error_msg += "\n".join(f"  - {issue}" for issue in issues)
                    return False, error_msg
                
                if self.verbose:
                    print(f"   ✅ Validation passed")
            
            # Write atomically with backup
            if self.verbose:
                print(f"\n💾 Saving modified paper...")
            
            success, error = SafeFileWriter.write_atomic(
                paper_path,
                modified_content,
                create_backup=self.create_backups,
                validate_before_save=self.validate_output
            )
            
            if not success:
                return False, f"Failed to save: {error}"
            
            # Success!
            modified_lines = len(modified_content.split('\n'))
            modified_size = len(modified_content)
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"✅ Paper modification successful!")
                print(f"{'='*70}")
                print(f"Original: {len(original_content.split('\n')):,} lines, {len(original_content):,} bytes")
                print(f"Modified: {modified_lines:,} lines, {modified_size:,} bytes")
                print(f"File: {paper_path}")
                if self.create_backups:
                    print(f"Backup: Created in backups/ folder")
                print(f"{'='*70}\n")
            
            return True, None
            
        except Exception as e:
            return False, f"Modification failed: {e}"
    
    def _modify_with_chunking(
        self,
        paper_path: Path,
        original_content: str,
        modification_instructions: str,
        system_prompt: Optional[str],
        **llm_kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Modification with document chunking for very large papers."""
        
        if self.verbose:
            print(f"\n📦 Using chunked modification strategy...")
        
        # Split document
        chunk_size = self.config.get("chunking", {}).get("chunk_size", 300)
        overlap = self.config.get("chunking", {}).get("overlap", 50)
        
        chunks = DocumentChunker.split_latex_document(
            original_content,
            max_lines_per_chunk=chunk_size,
            overlap_lines=overlap
        )
        
        if self.verbose:
            print(f"   Split into {len(chunks)} chunks")
        
        # For now, use direct modification even for large docs
        # (Chunking requires more sophisticated merge strategy)
        if self.verbose:
            print(f"   ℹ️  Using direct modification with unlimited output")
            print(f"      (sophisticated chunking merge not yet implemented)")
        
        return self._modify_direct(
            paper_path,
            original_content,
            modification_instructions,
            system_prompt,
            **llm_kwargs
        )


def modify_paper_safe(
    paper_path: str,
    instructions: str,
    model: str = "gpt-5-pro",
    verbose: bool = True,
    **kwargs
) -> Tuple[bool, Optional[str]]:
    """
    Convenience function for safe paper modification.
    
    Args:
        paper_path: Path to paper file
        instructions: Modification instructions
        model: Model to use
        verbose: Whether to print progress
        **kwargs: Additional arguments
        
    Returns:
        (success, error_message)
    """
    modifier = SafePaperModifier(
        model=model,
        verbose=verbose,
        enable_chunking=True,
        validate_output=True,
        create_backups=True
    )
    
    return modifier.modify_paper(
        Path(paper_path),
        instructions,
        **kwargs
    )


def test_safe_modification():
    """Test safe paper modification."""
    print("\n" + "="*70)
    print("🧪 Testing Safe Paper Modification")
    print("="*70 + "\n")
    
    # Create test paper
    test_paper = Path("test_paper.tex")
    test_content = r"""\documentclass{article}
\begin{document}
\section{Introduction}
This is a test paper.
\section{Methods}
Testing methods here.
\section{Conclusion}
Test conclusions.
\end{document}
"""
    
    test_paper.write_text(test_content, encoding='utf-8')
    print(f"📄 Created test paper: {test_paper}")
    
    # Test modification
    success, error = modify_paper_safe(
        str(test_paper),
        "Add a new section called 'Results' between Methods and Conclusion with some sample content.",
        model="gpt-4o",  # Use cheaper model for testing
        verbose=True
    )
    
    if success:
        print(f"\n✅ Modification successful!")
        modified = test_paper.read_text(encoding='utf-8')
        print(f"\nModified content:\n{'-'*70}\n{modified}\n{'-'*70}")
    else:
        print(f"\n❌ Modification failed: {error}")
    
    # Cleanup
    test_paper.unlink()
    backup_dir = test_paper.parent / "backups"
    if backup_dir.exists():
        import shutil
        shutil.rmtree(backup_dir)
    
    print(f"\n🧹 Cleaned up test files")
    print("\n" + "="*70)
    print("✅ Safe Modification Test Complete")
    print("="*70 + "\n")


# Export main functions
__all__ = [
    'SafePaperModifier',
    'modify_paper_safe',
    'test_safe_modification',
]
