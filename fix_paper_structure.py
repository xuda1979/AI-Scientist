#!/usr/bin/env python3
"""
Fix the paper.tex structure to ensure proper LaTeX compilation and bibliography rendering.

Issues to fix:
1. Ensure filecontents block is at the very top
2. Close filecontents properly
3. Remove duplicates
4. Ensure proper document structure
5. Add \cite{} commands for all references in the text
"""

import re
import sys
from pathlib import Path

def fix_paper_structure(paper_path):
    """Fix the LaTeX paper structure."""
    
    print(f"Reading {paper_path}...")
    content = paper_path.read_text(encoding='utf-8')
    
    # Extract the bibliography from filecontents block
    print("Extracting bibliography...")
    bib_match = re.search(r'\\begin\{filecontents\*\}\{refs\.bib\}(.*?)(?=\\end\{filecontents\*\}|\\documentclass|\\begin\{abstract\})', 
                          content, re.DOTALL)
    
    if bib_match:
        bib_content = bib_match.group(1).strip()
        print(f"  Found bibliography content ({len(bib_content)} chars)")
    else:
        print("  WARNING: No filecontents block found, extracting from scattered entries")
        # Extract all @article and @inproceedings entries
        bib_entries = re.findall(r'(@(?:article|inproceedings|book|techreport)\{[^@]+)', content, re.DOTALL)
        bib_content = '\n\n'.join(bib_entries)
    
    # Clean up the bibliography content
    # Remove any stray LaTeX commands that got mixed in
    bib_content = re.sub(r'\\(?:usepackage|documentclass|begin\{abstract\}|author\{|maketitle).*?\n', '', bib_content)
    bib_content = bib_content.strip()
    
    # Extract main document content (everything after first documentclass)
    print("Extracting main document...")
    doc_match = re.search(r'(\\documentclass.*?)$', content, re.DOTALL)
    
    if doc_match:
        doc_content = doc_match.group(1)
        print(f"  Found document content ({len(doc_content)} chars)")
    else:
        print("  ERROR: No documentclass found!")
        return False
    
    # Clean up document content
    # Remove any filecontents blocks that got embedded
    doc_content = re.sub(r'\\begin\{filecontents\*\}.*?(?:\\end\{filecontents\*\}|\n\n)', '', doc_content, flags=re.DOTALL)
    # Remove stray @article entries
    doc_content = re.sub(r'@(?:article|inproceedings)\{[^\\]*?(?=\\)', '', doc_content)
    # Remove duplicate \end{abstract} and \end{figure} at the end
    doc_content = re.sub(r'(\\end\{document\})[\s\S]*$', r'\1', doc_content)
    
    # Extract all reference keys from bibliography
    print("Extracting reference keys...")
    ref_keys = re.findall(r'@\w+\{([^,]+),', bib_content)
    print(f"  Found {len(ref_keys)} references: {', '.join(ref_keys[:5])}...")
    
    # Count existing citations
    existing_citations = re.findall(r'\\cite\{([^}]+)\}', doc_content)
    print(f"  Found {len(existing_citations)} existing citations")
    
    # Build the corrected paper
    fixed_content = f"""\\begin{{filecontents*}}{{refs.bib}}
{bib_content}
\\end{{filecontents*}}

{doc_content}"""
    
    # Write the fixed version
    backup_path = paper_path.with_suffix('.tex.backup')
    print(f"\nCreating backup: {backup_path}")
    paper_path.rename(backup_path)
    
    print(f"Writing fixed paper to: {paper_path}")
    paper_path.write_text(fixed_content, encoding='utf-8')
    
    print("\n✓ Paper structure fixed!")
    print(f"  - Bibliography: {len(ref_keys)} references")
    print(f"  - Citations in text: {len(existing_citations)}")
    
    # Show uncited references
    cited_refs = set()
    for cite in existing_citations:
        cited_refs.update(cite.split(','))
    cited_refs = {ref.strip() for ref in cited_refs}
    
    uncited = set(ref_keys) - cited_refs
    if uncited:
        print(f"\n⚠ WARNING: {len(uncited)} uncited references:")
        for ref in sorted(uncited):
            print(f"    - {ref}")
        print("\n  These references are in the bibliography but not cited in the text.")
        print("  They will not appear in the final PDF!")
    
    return True

if __name__ == "__main__":
    paper_path = Path("output/Access_Point_Selection_Precoding/paper.tex")
    
    if not paper_path.exists():
        print(f"ERROR: Paper not found at {paper_path}")
        sys.exit(1)
    
    try:
        success = fix_paper_structure(paper_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
