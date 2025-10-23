from __future__ import annotations

import pytest

import importlib.util
import sys
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "comprehensive_workflow_test_module",
    Path(__file__).resolve().parents[1] / "src" / "core" / "comprehensive_workflow.py",
)
assert SPEC and SPEC.loader
_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _MODULE
SPEC.loader.exec_module(_MODULE)  # type: ignore[arg-type]

parse_stage_payload = _MODULE.parse_stage_payload
WorkflowParsingError = _MODULE.WorkflowParsingError


def test_parse_stage_payload_from_fenced_block() -> None:
    payload = parse_stage_payload(
        """
        Here is the plan:
        ```json
        {"topics": ["Topic A"], "literature_trends": []}
        ```
        """
    )
    assert payload["topics"] == ["Topic A"]


def test_parse_stage_payload_without_fence() -> None:
    payload = parse_stage_payload('{"key": "value", "items": [1, 2]}')
    assert payload["items"] == [1, 2]


def test_parse_stage_payload_invalid_json() -> None:
    with pytest.raises(WorkflowParsingError):
        parse_stage_payload("{" "invalid" "}")


def test_parse_stage_payload_no_json_object() -> None:
    with pytest.raises(WorkflowParsingError):
        parse_stage_payload("No JSON here")
