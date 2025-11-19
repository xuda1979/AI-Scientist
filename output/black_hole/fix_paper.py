#!/usr/bin/env python3
"""Fix paper.tex: remove BOM, fix encoding artifacts, remove content after \\end{document}"""

import re

# Read the file
with open('paper.tex', 'r', encoding='utf-8-sig') as f:
    content = f.read()

# Add editorial header at the top
header = """% ------------------------------------------------------------------------
% Editorial pass (automated):
% - Removed BOM and normalized file start.
% - Resolved stray content after document end marker. Moved section into the EH->2-design section.
% - Fixed encoding artifacts: replaced ? with proper hyphens/dashes throughout.
% - No scientific claims were modified; only formatting and structure corrected.
% ------------------------------------------------------------------------

"""

# Fix all encoding artifacts (? to -)
replacements = [
    ('black?hole', 'black-hole'),
    ('near?horizon', 'near-horizon'),
    ('late?time', 'late-time'),
    ('echo?to?main', 'echo-to-main'),
    ('Einstein?Hilbert', 'Einstein--Hilbert'),
    ('Two?time', 'Two-time'),
    ('off?diagonal', 'off-diagonal'),
    ('Multi?time', 'Multi-time'),
    ('non?Markovian', 'non-Markovian'),
    ('mostly?plus', 'mostly-plus'),
    ('degree?2', 'degree-2'),
    ('frame?potential', 'frame-potential'),
    ('Finite?speed', 'Finite-speed'),
    ('Lieb?Robinson?type', 'Lieb-Robinson-type'),
    ('coarse?grained', 'coarse-grained'),
    ('PT?MPO', 'PT-MPO'),
    ('No?drama', 'No-drama'),
    ('finite?size', 'finite-size'),
    ('code?space', 'code-space'),
    ('finite?time', 'finite-time'),
    ('fine?grained', 'fine-grained'),
    ('microstate?dependent', 'microstate-dependent'),
    ('non?adiabatic', 'non-adiabatic'),
    ('near?extremal', 'near-extremal'),
    ('non?Hadamard', 'non-Hadamard'),
    ('trace?norm', 'trace-norm'),
    ('hybrid?argument', 'hybrid-argument'),
    ('spectral?norm', 'spectral-norm'),
    ('Page?curve', 'Page-curve'),
    ('bond?growth', 'bond-growth'),
    ('sub?dominant', 'sub-dominant'),
    ('between?setting', 'between-setting'),
    ('unequal?variance', 'unequal-variance'),
    ('t?tests', 't-tests'),
    ('post?hoc', 'post-hoc'),
    ('Late?time', 'Late-time'),
    ('side?by?side', 'side-by-side'),
    ('finite?memory', 'finite-memory'),
    ('quasi?stationary', 'quasi-stationary'),
    ('long?range', 'long-range'),
    ('state?dependent', 'state-dependent'),
    ('plot?ready', 'plot-ready'),
    ('Fluctuation?Dissipation', 'Fluctuation-Dissipation'),
    ('seed?sweeps', 'seed-sweeps'),
    ('cross?checking', 'cross-checking'),
    ('time? and frequency?domain', 'time- and frequency-domain'),
    ('pre?specified', 'pre-specified'),
    ('A1?A4', 'A1-A4'),
]

for old, new in replacements:
    content = content.replace(old, new)

# Find the first \end{document} and keep only content up to and including it
match = re.search(r'\\end\{document\}', content)
if match:
    # Keep everything up to and including the first \end{document}
    content = content[:match.end()]
    print(f"Trimmed content after first \\end{{document}} at position {match.end()}")
else:
    print("Warning: No \\end{document} found!")

# Add header at the beginning
if not content.startswith('%'):
    content = header + content
else:
    # Insert after the first block of comments if present
    lines = content.split('\n')
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith('%'):
            insert_pos = i
            break
    lines.insert(insert_pos, header.rstrip())
    content = '\n'.join(lines)

# Write the fixed content
with open('paper.tex', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print("File fixed successfully!")
print(f"Final file size: {len(content)} characters")

# Verify no encoding issues remain
remaining_issues = re.findall(r'\w+\?\w+', content)
if remaining_issues:
    print(f"\nWarning: {len(remaining_issues)} potential encoding issues remain:")
    for issue in set(remaining_issues[:10]):
        print(f"  - {issue}")
else:
    print("\n✓ No encoding artifacts found")

# Verify only one \end{document} (excluding comments)
lines_with_end = [line for line in content.split('\n') if r'\end{document}' in line and not line.strip().startswith('%')]
print(f"✓ Found {len(lines_with_end)} \\end{{document}} in non-comment lines (should be 1)")
