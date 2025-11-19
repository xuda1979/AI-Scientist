
from __future__ import annotations
import json, textwrap, types, time, multiprocessing as mp
from typing import List, Dict, Any

def load_humaneval_jsonl(path: str, max_items: int | None = None) -> List[Dict[str, Any]]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                break
            out.append(json.loads(line))
    return out

def _exec_with_timeout(code: str, test: str, timeout: float = 2.0) -> bool:
    """Execute generated code + unit test in a subprocess with a timeout.
    Returns True on pass, False otherwise. This is a minimal sandbox; harden for production.
    """
    def worker(pipe):
        try:
            ns: Dict[str, Any] = {}
            exec(code, ns, ns)  # define function(s)
            exec(test, ns, ns)  # run tests (should raise on failure)
            pipe.send(True)
        except Exception:
            pipe.send(False)

    parent, child = mp.Pipe()
    p = mp.Process(target=worker, args=(child,))
    p.start()
    parent_conn = parent
    parent_conn.poll(timeout)
    result = False
    try:
        if parent_conn.poll(timeout):
            result = parent_conn.recv()
    finally:
        if p.is_alive():
            p.terminate()
        p.join()
    return bool(result)

def humaneval_verify(code: str, test: str) -> bool:
    return _exec_with_timeout(code, test, timeout=2.0)
