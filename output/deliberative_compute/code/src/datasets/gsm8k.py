
from __future__ import annotations
import json
from typing import List, Dict, Any, Iterable

def load_gsm8k_jsonl(path: str, max_items: int | None = None) -> List[Dict[str, Any]]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                break
            out.append(json.loads(line))
    return out

def simple_math_verifier(answer_text: str) -> bool:
    """Very light-weight verifier: looks for a final integer or decimal and returns True if parsable.
    Replace with a real checker for GSM8K (e.g., normalize final numeric answer vs label)."""
    import re
    m = re.findall(r"[-+]?[0-9]*\.?[0-9]+", answer_text)
    return len(m) > 0
