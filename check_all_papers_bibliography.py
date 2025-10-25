#!/usr/bin/env python3
"""Check all papers for bibliography presence."""

import os
import re
from pathlib import Path

papers_dir = Path(r'C:\Users\Lenovo\papers')

print("=" * 100)
print("CHECKING ALL PAPERS FOR BIBLIOGRAPHY")
print("=" * 100)

for paper_folder in sorted(papers_dir.iterdir()):
    if not paper_folder.is_dir():
        continue
    
    paper_tex = paper_folder / 'paper.tex'
    if not paper_tex.exists():
        continue
    
    print(f"\n{paper_folder.name}:")
    
    try:
        with open(paper_tex, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Count citations
        citations = re.findall(r'\\cite\{[^}]+\}', content)
        num_citations = len(citations)
        
        # Check for bibliography commands
        has_bibliography_cmd = bool(re.search(r'\\bibliography\{[^}]+\}', content))
        has_bibliographystyle = bool(re.search(r'\\bibliographystyle\{[^}]+\}', content))
        has_thebibliography = bool(re.search(r'\\begin\{thebibliography\}', content))
        
        # Check refs.bib exists
        refs_bib = paper_folder / 'refs.bib'
        has_refs_file = refs_bib.exists()
        if has_refs_file:
            refs_size = refs_bib.stat().st_size
        else:
            refs_size = 0
        
        # Report
        print(f"  Citations: {num_citations}")
        print(f"  \\bibliography{{}} command: {'YES' if has_bibliography_cmd else 'NO'}")
        print(f"  \\bibliographystyle{{}} command: {'YES' if has_bibliographystyle else 'NO'}")
        print(f"  \\begin{{thebibliography}}: {'YES' if has_thebibliography else 'NO'}")
        print(f"  refs.bib file: {'YES (' + str(refs_size) + ' bytes)' if has_refs_file else 'NO'}")
        
        # Flag issues
        if num_citations > 0 and not (has_bibliography_cmd or has_thebibliography):
            print(f"  ⚠️  CRITICAL: Has {num_citations} citations but NO bibliography section!")
        elif num_citations > 0 and has_bibliography_cmd and not has_bibliographystyle:
            print(f"  ⚠️  WARNING: Has \\bibliography but missing \\bibliographystyle")
        elif has_bibliography_cmd and not has_refs_file:
            print(f"  ⚠️  WARNING: Has \\bibliography command but no refs.bib file")
        elif num_citations > 0:
            print(f"  ✅ Bibliography appears OK")
        
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 100)
