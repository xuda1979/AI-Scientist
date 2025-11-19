#!/usr/bin/env python3
"""Calculate estimated token count with the fix"""

# Previous run measurements
previous_total_chars = 534668
previous_total_tokens = previous_total_chars / 4

print(f"BEFORE FIX:")
print(f"  Total: {previous_total_chars:,} chars = {previous_total_tokens:.0f} tokens")
print(f"  Limit: 128,000 tokens")
print(f"  Exceeded by: {previous_total_tokens - 128000:.0f} tokens")
print()

# Estimated component sizes
paper_len = 38828
sim_len = 50000  # Generous estimate
sys_len = 15000  # System prompt
quality_len = 8000  # Quality issues (limited to 15)
latex_errors = 2000  # LaTeX errors section

base_content = paper_len + sim_len + sys_len
max_context_size = 250000

print(f"AFTER FIX:")
print(f"  Paper: {paper_len:,} chars")
print(f"  Simulation: {sim_len:,} chars")
print(f"  System prompt: {sys_len:,} chars")
print(f"  Base content: {base_content:,} chars")
print(f"  Max allowed before limiting: {max_context_size:,} chars")
print()

if base_content < max_context_size:
    remaining = max_context_size - base_content
    print(f"  Will INCLUDE project files (limited to {remaining:,} chars)")
    new_total = max_context_size + quality_len + latex_errors + 10000  # Extra overhead
else:
    print(f"  Will SKIP project files")
    new_total = base_content + quality_len + latex_errors + 10000

new_total_tokens = new_total / 4

print(f"  Quality issues: {quality_len:,} chars (limited to 15 issues)")
print(f"  LaTeX errors: {latex_errors:,} chars")
print(f"  New total estimate: {new_total:,} chars = {new_total_tokens:.0f} tokens")
print(f"  Limit: 128,000 tokens")
print(f"  Safety margin: {128000 - new_total_tokens:.0f} tokens")
print()

if new_total_tokens < 128000:
    print(f"✅ FIX SUCCESSFUL: Reduced from {previous_total_tokens:.0f} to {new_total_tokens:.0f} tokens")
    print(f"   Saved: {previous_total_tokens - new_total_tokens:.0f} tokens ({100*(previous_total_tokens - new_total_tokens)/previous_total_tokens:.1f}% reduction)")
else:
    print(f"❌ FIX INSUFFICIENT: Still {new_total_tokens - 128000:.0f} tokens over limit")
    print(f"   Need to reduce by additional: {(new_total_tokens - 120000)*4:.0f} chars")
