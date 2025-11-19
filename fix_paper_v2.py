#!/usr/bin/env python3
"""
Properly fix the paper.tex structure by extracting and reorganizing all components.
"""

import re
from pathlib import Path

def extract_all_bib_entries(content):
    """Extract all @article, @inproceedings, etc. entries."""
    # Find all bibliography entries
    pattern = r'(@(?:article|inproceedings|book|techreport|misc)\{[a-zA-Z0-9_]+,[\s\S]*?\n\})'
    entries = re.findall(pattern, content)
    
    # Deduplicate by key
    seen_keys = set()
    unique_entries = []
    for entry in entries:
        key_match = re.search(r'@\w+\{([^,]+),', entry)
        if key_match:
            key = key_match.group(1)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_entries.append(entry)
    
    return unique_entries, sorted(seen_keys)

def extract_document_content(content):
    """Extract everything from \\documentclass onwards, cleaning out embedded bib entries."""
    # Find documentclass
    doc_start = content.find('\\documentclass')
    if doc_start == -1:
        raise ValueError("No \\documentclass found!")
    
    doc_content = content[doc_start:]
    
    # Remove any embedded bib entries
    doc_content = re.sub(r'@(?:article|inproceedings|book|techreport|misc)\{[a-zA-Z0-9_]+,[\s\S]*?\n\}', '', doc_content)
    
    # Remove stray lines from filecontents block
    doc_content = re.sub(r'\\begin\{filecontents\*\}\{refs\.bib\}', '', doc_content)
    doc_content = re.sub(r'\\end\{filecontents\*\}', '', doc_content)
    
    # Clean up duplicate end tags at the very end
    # First, find the last \end{document}
    last_end_doc = doc_content.rfind('\\end{document}')
    if last_end_doc != -1:
        doc_content = doc_content[:last_end_doc + len('\\end{document}')]
    
    return doc_content.strip()

def main():
    paper_path = Path("output/Access_Point_Selection_Precoding/paper.tex")
    
    print(f"Reading {paper_path}...")
    content = paper_path.read_text(encoding='utf-8')
    
    # Extract all bibliography entries
    print("Extracting bibliography entries...")
    bib_entries, bib_keys = extract_all_bib_entries(content)
    print(f"  Found {len(bib_entries)} unique entries")
    print(f"  Keys: {', '.join(bib_keys[:10])}...")
    
    # Extract document content
    print("Extracting document content...")
    doc_content = extract_document_content(content)
    print(f"  Document content: {len(doc_content)} characters")
    
    # Count citations in document
    citations = re.findall(r'\\cite\{([^}]+)\}', doc_content)
    cited_keys = set()
    for cite in citations:
        cited_keys.update([k.strip() for k in cite.split(',')])
    
    print(f"  Found {len(citations)} \\cite commands citing {len(cited_keys)} unique keys")
    
    # Build the corrected paper
    print("Building corrected paper...")
    bib_block = '\n\n'.join(bib_entries)
    
    fixed_content = f"""\\begin{{filecontents*}}{{refs.bib}}
{bib_block}
\\end{{filecontents*}}

{doc_content}
"""
    
    # Create backup
    backup_path = paper_path.with_suffix('.tex.broken')
    print(f"Creating backup: {backup_path}")
    paper_path.rename(backup_path)
    
    # Write fixed version
    print(f"Writing fixed paper...")
    paper_path.write_text(fixed_content, encoding='utf-8')
    
    print("\n✅ Paper structure fixed!")
    print(f"  - Bibliography: {len(bib_entries)} entries ({', '.join(bib_keys[:5])}...)")
    print(f"  - Citations: {len(cited_keys)} unique keys cited")
    
    # Check for uncited references
    uncited = set(bib_keys) - cited_keys
    if uncited:
        print(f"\n⚠️  {len(uncited)} UNCITED REFERENCES (will not appear in PDF):")
        for key in sorted(uncited)[:10]:
            print(f"    - {key}")
        if len(uncited) > 10:
            print(f"    ... and {len(uncited) - 10} more")
    
    # Check for missing references
    missing = cited_keys - set(bib_keys)
    if missing:
        print(f"\n❌ {len(missing)} MISSING REFERENCES (cited but not in bibliography):")
        for key in sorted(missing)[:10]:
            print(f"    - {key}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

if __name__ == "__main__":
    main()
