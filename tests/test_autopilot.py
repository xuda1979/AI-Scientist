#!/usr/bin/env python3
"""
Tests for the autopilot module.
Run: python -m pytest tests/test_autopilot.py -v
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import sys

# Ensure the repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autopilot.config import AutoPilotConfig, StageConfig, STAGES
from autopilot.workspace import WorkspaceManager
from autopilot.prompt_optimizer import PromptOptimizer, STAGE_DESCRIPTIONS
from autopilot.quality_gate import QualityGate
from autopilot.stage_executor import StageExecutor
from autopilot.agent import AutoPilotAgent


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with sample files."""
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "paper.tex").write_text(
        r"""\documentclass{article}
\title{Test Paper}
\author{Test Author}
\begin{document}
\maketitle
\begin{abstract}
This is a test abstract.
\end{abstract}
\section{Introduction}
Hello world.
\section{Conclusion}
Done.
\end{document}
""",
        encoding="utf-8",
    )
    (project / "simulation.py").write_text(
        "import numpy as np\nprint('hello')\n", encoding="utf-8"
    )
    (project / "notes.md").write_text("# Research Notes\n- idea 1\n", encoding="utf-8")
    return project


@pytest.fixture
def mock_llm():
    """A mock LLM function that returns a fixed response."""

    def _llm(messages, model="test", prompt_type="general", **kwargs):
        return (
            "Here is my analysis.\n\n"
            "```file_actions\n"
            '[{"action": "write", "path": "results.txt", "content": "result data"}]\n'
            "```"
        )

    return _llm


@pytest.fixture
def config(tmp_project):
    return AutoPilotConfig(
        topic="Test Topic",
        field_name="Computer Science",
        research_question="Does X improve Y?",
        project_dir=str(tmp_project),
        model="test-model",
        max_global_iterations=1,
        prompt_candidates_per_stage=1,
        use_meta_prompt=False,
        save_trace=False,
    )


# ── Config tests ─────────────────────────────────────────────────────────

class TestAutoConfig:
    def test_default_stages(self):
        cfg = AutoPilotConfig()
        assert len(cfg.stages) == len(STAGES)
        assert cfg.stages[0] == "literature_review"

    def test_stage_config_defaults(self):
        cfg = AutoPilotConfig()
        sc = cfg.get_stage_config("literature_review")
        assert isinstance(sc, StageConfig)
        assert sc.max_attempts == 3
        assert sc.temperature == 0.4

    def test_stage_config_overrides(self):
        cfg = AutoPilotConfig(
            stage_overrides={"initial_draft": {"temperature": 0.9, "max_attempts": 5}}
        )
        sc = cfg.get_stage_config("initial_draft")
        assert sc.temperature == 0.9
        assert sc.max_attempts == 5

    def test_save_and_load(self, tmp_path):
        cfg = AutoPilotConfig(topic="Save Test", model="gpt-4o")
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = AutoPilotConfig.from_file(path)
        assert loaded.topic == "Save Test"
        assert loaded.model == "gpt-4o"


# ── Workspace tests ──────────────────────────────────────────────────────

class TestWorkspaceManager:
    def test_list_files(self, tmp_project):
        ws = WorkspaceManager(tmp_project)
        files = ws.list_files()
        assert "paper.tex" in files
        assert "simulation.py" in files
        assert "notes.md" in files

    def test_read_file(self, tmp_project):
        ws = WorkspaceManager(tmp_project)
        content = ws.read_file("paper.tex")
        assert "\\title{Test Paper}" in content

    def test_write_file_new(self, tmp_project):
        ws = WorkspaceManager(tmp_project, create_backups=False)
        result = ws.write_file("new_file.txt", "hello world")
        assert "OK" in result
        assert (tmp_project / "new_file.txt").read_text() == "hello world"

    def test_patch_file(self, tmp_project):
        ws = WorkspaceManager(tmp_project, create_backups=False)
        result = ws.patch_file("notes.md", "idea 1", "idea 1 (updated)")
        assert "OK" in result
        assert "updated" in ws.read_file("notes.md")

    def test_content_protection_blocks(self, tmp_project):
        ws = WorkspaceManager(tmp_project, content_protection=True, max_word_loss_pct=10)
        # Try to replace paper.tex with much shorter content
        result = ws.write_file("paper.tex", "short")
        assert "REJECTED" in result

    def test_snapshot(self, tmp_project):
        ws = WorkspaceManager(tmp_project)
        snap = ws.snapshot()
        assert "FILE: paper.tex" in snap
        assert "\\title{Test Paper}" in snap

    def test_directory_traversal_blocked(self, tmp_project):
        ws = WorkspaceManager(tmp_project)
        with pytest.raises(PermissionError):
            ws.read_file("../../etc/passwd")

    def test_apply_actions(self, tmp_project):
        ws = WorkspaceManager(tmp_project, create_backups=False)
        actions = [
            {"action": "write", "path": "data.csv", "content": "a,b\n1,2\n"},
            {"action": "patch", "path": "notes.md", "search": "idea 1", "replace": "idea X"},
        ]
        results = ws.apply_actions(actions)
        assert all("OK" in r for r in results)

    def test_parse_file_actions_json(self):
        response = (
            "Here is my work.\n\n"
            "```file_actions\n"
            '[{"action": "write", "path": "test.py", "content": "print(1)"}]\n'
            "```"
        )
        actions = WorkspaceManager.parse_file_actions_from_response(response)
        assert len(actions) == 1
        assert actions[0]["path"] == "test.py"

    def test_parse_file_actions_blocks(self):
        response = (
            "=== FILE: test.py ===\nprint(1)\n\n"
            "=== FILE: data.txt ===\nhello\n"
        )
        actions = WorkspaceManager.parse_file_actions_from_response(response)
        assert len(actions) == 2


# ── Prompt optimizer tests ───────────────────────────────────────────────

class TestPromptOptimizer:
    def test_static_prompt(self, config, mock_llm):
        opt = PromptOptimizer(config, mock_llm)
        prompt = opt.best_prompt("literature_review", "(empty project)")
        assert "Literature Review" in prompt
        assert "Test Topic" in prompt

    def test_all_stages_have_descriptions(self):
        for stage in STAGES:
            assert stage in STAGE_DESCRIPTIONS, f"Missing description for {stage}"


# ── Quality gate tests ───────────────────────────────────────────────────

class TestQualityGate:
    def test_heuristic_passes_good_paper(self, tmp_project):
        ws = WorkspaceManager(tmp_project)
        snapshot = ws.snapshot()

        def mock_llm(messages, **kw):
            return (
                '```json\n'
                '{"scores": {"completeness": 0.9, "quality": 0.8, "correctness": 0.9, "integration": 0.8}, '
                '"overall": 0.85, "issues": [], "suggestions": []}\n'
                '```'
            )

        qg = QualityGate(mock_llm, "test", quality_threshold=0.7)
        result = qg.evaluate("initial_draft", "good output", snapshot, [])
        assert result["pass"] is True
        assert result["verdict"] == "proceed"

    def test_heuristic_fails_missing_file(self):
        def mock_llm(messages, **kw):
            return '```json\n{"overall": 0.5, "issues": ["bad"], "suggestions": []}\n```'

        qg = QualityGate(mock_llm, "test", quality_threshold=0.7)
        result = qg.evaluate("initial_draft", "", "(empty project)", [])
        assert result["score"] < 0.7


# ── Stage executor tests ────────────────────────────────────────────────

class TestStageExecutor:
    def test_execute_success(self, config, tmp_project, mock_llm):
        ws = WorkspaceManager(tmp_project, create_backups=False)
        se = StageExecutor(config, ws, mock_llm)
        stage_cfg = StageConfig(enable_self_reflection=False)
        result = se.execute("literature_review", "Do a review", stage_cfg)
        assert result["success"] is True
        assert len(result["file_actions"]) == 1
        assert (tmp_project / "results.txt").exists()

    def test_execute_llm_failure(self, config, tmp_project):
        def failing_llm(*args, **kwargs):
            raise RuntimeError("API error")

        ws = WorkspaceManager(tmp_project, create_backups=False)
        se = StageExecutor(config, ws, failing_llm)
        stage_cfg = StageConfig()
        result = se.execute("literature_review", "Do a review", stage_cfg)
        assert result["success"] is False
        assert "API error" in result["error"]


# ── Agent integration tests ──────────────────────────────────────────────

class TestAutoPilotAgent:
    def test_run_minimal(self, tmp_project):
        """Run the agent with a mock LLM and a single stage."""
        call_count = 0

        def counting_llm(messages, **kw):
            nonlocal call_count
            call_count += 1
            # Return a response that writes a paper.tex
            if call_count <= 2:  # stage execution + possible self-reflection
                return (
                    "Here is the literature review.\n\n"
                    "```file_actions\n"
                    '[{"action": "write", "path": "lit_review.md", "content": "# Literature Review\\n\\nKey papers..."}]\n'
                    "```"
                )
            else:
                # Quality evaluation
                return (
                    '```json\n'
                    '{"overall": 0.9, "issues": [], "suggestions": []}\n'
                    '```'
                )

        config = AutoPilotConfig(
            topic="Test Topic",
            field_name="CS",
            research_question="Test?",
            project_dir=str(tmp_project),
            model="test",
            stages=["literature_review"],
            max_global_iterations=1,
            prompt_candidates_per_stage=1,
            use_meta_prompt=False,
            save_trace=False,
        )

        agent = AutoPilotAgent(config, llm_fn=counting_llm)
        result = agent.run()

        assert result["status"] == "completed"
        assert result["stages_completed"] >= 1
        assert (tmp_project / "lit_review.md").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
