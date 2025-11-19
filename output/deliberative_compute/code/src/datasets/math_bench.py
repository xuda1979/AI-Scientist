
from __future__ import annotations
import json
from typing import List, Dict, Any

def load_math_jsonl(path: str, max_items: int | None = None) -> List[Dict[str, Any]]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                break
            out.append(json.loads(line))
    return out
