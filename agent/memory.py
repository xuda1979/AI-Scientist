"""
AgentMemory — persistent state, conversation history, and knowledge base.

Tracks:
- Conversation history per stage (with compression for long contexts)
- Structured research state (hypotheses, findings, results)
- Knowledge base (papers found, key insights)
- Decision log (why the agent chose each path)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperReference:
    """A reference to an academic paper."""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    abstract: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    relevance_score: float = 0.0
    key_findings: str = ""
    bibtex: str = ""


@dataclass
class Hypothesis:
    """A research hypothesis tracked through the pipeline."""
    id: str = ""
    statement: str = ""
    rationale: str = ""
    status: str = "proposed"  # proposed | testing | supported | refuted | revised
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_id: str = ""
    description: str = ""
    code_file: str = ""
    status: str = "pending"  # pending | running | completed | failed
    metrics: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timestamp: str = ""


@dataclass
class ResearchState:
    """Structured state of the research project."""
    # Research phase
    literature: List[PaperReference] = field(default_factory=list)
    research_gaps: List[str] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)

    # Design phase
    methodology: str = ""
    experiment_plan: str = ""

    # Implementation phase
    experiments: List[ExperimentResult] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)

    # Writing phase
    outline: str = ""
    current_draft_summary: str = ""
    figures: List[str] = field(default_factory=list)

    # Review phase
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    revision_history: List[str] = field(default_factory=list)

    # Meta
    current_phase: str = "research"
    current_stage: str = ""
    stages_completed: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)


class AgentMemory:
    """
    Persistent memory system for the autonomous research agent.

    Stores conversation history, research state, and knowledge base.
    Supports compression of old context to stay within token limits.
    """

    def __init__(
        self,
        memory_dir: Path | str = "agent_memory",
        max_history_per_stage: int = 10,
        max_context_tokens: int = 120000,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_per_stage = max_history_per_stage
        self.max_context_tokens = max_context_tokens

        # Core state
        self.state = ResearchState()
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.decision_log: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, Any] = {}

        # Load existing state if available
        self._load_state()

    # ── Conversation History ─────────────────────────────────────────────

    def add_message(self, stage: str, role: str, content: str) -> None:
        """Add a message to the conversation history for a stage."""
        if stage not in self.conversation_history:
            self.conversation_history[stage] = []

        self.conversation_history[stage].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

        # Trim if too long
        if len(self.conversation_history[stage]) > self.max_history_per_stage * 2:
            self._compress_history(stage)

    def get_history(self, stage: str) -> List[Dict[str, str]]:
        """Get conversation history for a stage."""
        return self.conversation_history.get(stage, [])

    def get_relevant_history(self, stage: str, max_messages: int = 6) -> List[Dict[str, str]]:
        """Get the most relevant recent history for a stage."""
        history = self.get_history(stage)
        if len(history) <= max_messages:
            return history
        # Keep first message (system context) + last N messages
        return history[:1] + history[-(max_messages - 1):]

    def get_cross_stage_context(self, current_stage: str) -> str:
        """
        Build a summary of key information from previous stages
        that's relevant to the current stage.
        """
        context_parts = []

        # Always include research state summary
        context_parts.append(self._state_summary())

        # Include key decisions
        recent_decisions = self.decision_log[-5:] if self.decision_log else []
        if recent_decisions:
            decisions_text = "\n".join(
                f"- [{d.get('stage', '?')}] {d.get('decision', '')}: {d.get('reason', '')}"
                for d in recent_decisions
            )
            context_parts.append(f"## Recent Decisions\n{decisions_text}")

        return "\n\n".join(context_parts)

    # ── Research State ───────────────────────────────────────────────────

    def add_paper(self, paper: PaperReference) -> None:
        """Add a paper to the literature collection."""
        # Avoid duplicates
        existing_titles = {p.title.lower() for p in self.state.literature}
        if paper.title.lower() not in existing_titles:
            self.state.literature.append(paper)
            logger.info("Added paper: %s", paper.title[:80])

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add or update a hypothesis."""
        for i, h in enumerate(self.state.hypotheses):
            if h.id == hypothesis.id:
                self.state.hypotheses[i] = hypothesis
                return
        self.state.hypotheses.append(hypothesis)

    def add_experiment(self, result: ExperimentResult) -> None:
        """Record an experiment result."""
        result.timestamp = datetime.now().isoformat()
        self.state.experiments.append(result)

    def add_finding(self, finding: str) -> None:
        """Record a key research finding."""
        if finding not in self.state.key_findings:
            self.state.key_findings.append(finding)

    def add_review(self, review: Dict[str, Any]) -> None:
        """Record a review."""
        review["timestamp"] = datetime.now().isoformat()
        self.state.reviews.append(review)

    def log_decision(self, stage: str, decision: str, reason: str, alternatives: List[str] = None) -> None:
        """Log a decision made by the agent."""
        self.decision_log.append({
            "stage": stage,
            "decision": decision,
            "reason": reason,
            "alternatives": alternatives or [],
            "timestamp": datetime.now().isoformat(),
        })

    def update_stage(self, stage: str, phase: str = "") -> None:
        """Mark a stage as the current stage."""
        self.state.current_stage = stage
        if phase:
            self.state.current_phase = phase
        if stage not in self.state.stages_completed:
            self.state.stages_completed.append(stage)

    def set_quality_score(self, stage: str, score: float) -> None:
        """Record a quality score for a stage."""
        self.state.quality_scores[stage] = score

    # ── Knowledge Base ───────────────────────────────────────────────────

    def store_knowledge(self, key: str, value: Any) -> None:
        """Store arbitrary knowledge in the knowledge base."""
        self.knowledge_base[key] = value

    def get_knowledge(self, key: str, default: Any = None) -> Any:
        """Retrieve knowledge from the knowledge base."""
        return self.knowledge_base.get(key, default)

    # ── Context Building ─────────────────────────────────────────────────

    def build_context_for_stage(self, stage: str) -> str:
        """
        Build comprehensive context for the LLM including:
        - Research state summary
        - Relevant cross-stage context
        - Key knowledge items
        """
        parts = []

        # 1. State summary
        parts.append(self._state_summary())

        # 2. Literature summary (if we have papers)
        if self.state.literature:
            lit_summary = self._literature_summary()
            parts.append(lit_summary)

        # 3. Experiment results summary
        if self.state.experiments:
            exp_summary = self._experiments_summary()
            parts.append(exp_summary)

        # 4. Recent reviews
        if self.state.reviews:
            review_summary = self._reviews_summary()
            parts.append(review_summary)

        # 5. Key findings
        if self.state.key_findings:
            findings = "\n".join(f"- {f}" for f in self.state.key_findings)
            parts.append(f"## Key Findings\n{findings}")

        return "\n\n".join(parts)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Save all memory state to disk."""
        try:
            # Save research state
            state_file = self.memory_dir / "research_state.json"
            state_file.write_text(
                json.dumps(asdict(self.state), indent=2, default=str),
                encoding="utf-8",
            )

            # Save conversation history
            history_file = self.memory_dir / "conversation_history.json"
            history_file.write_text(
                json.dumps(self.conversation_history, indent=2, default=str),
                encoding="utf-8",
            )

            # Save decision log
            decisions_file = self.memory_dir / "decision_log.json"
            decisions_file.write_text(
                json.dumps(self.decision_log, indent=2, default=str),
                encoding="utf-8",
            )

            # Save knowledge base
            kb_file = self.memory_dir / "knowledge_base.json"
            kb_file.write_text(
                json.dumps(self.knowledge_base, indent=2, default=str),
                encoding="utf-8",
            )

            logger.info("Memory saved to %s", self.memory_dir)
        except Exception as exc:
            logger.error("Failed to save memory: %s", exc)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load saved state from disk if available."""
        state_file = self.memory_dir / "research_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                # Reconstruct nested dataclasses
                if "literature" in data:
                    data["literature"] = [PaperReference(**p) for p in data["literature"]]
                if "hypotheses" in data:
                    data["hypotheses"] = [Hypothesis(**h) for h in data["hypotheses"]]
                if "experiments" in data:
                    data["experiments"] = [ExperimentResult(**e) for e in data["experiments"]]
                valid_fields = {f.name for f in ResearchState.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in valid_fields}
                self.state = ResearchState(**filtered)
                logger.info("Loaded research state from %s", state_file)
            except Exception as exc:
                logger.warning("Failed to load state: %s", exc)

        history_file = self.memory_dir / "conversation_history.json"
        if history_file.exists():
            try:
                self.conversation_history = json.loads(
                    history_file.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        decisions_file = self.memory_dir / "decision_log.json"
        if decisions_file.exists():
            try:
                self.decision_log = json.loads(
                    decisions_file.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        kb_file = self.memory_dir / "knowledge_base.json"
        if kb_file.exists():
            try:
                self.knowledge_base = json.loads(
                    kb_file.read_text(encoding="utf-8")
                )
            except Exception:
                pass

    def _compress_history(self, stage: str) -> None:
        """Compress old conversation history to save context window space."""
        history = self.conversation_history.get(stage, [])
        if len(history) <= self.max_history_per_stage:
            return

        # Keep first message + compress middle + keep last N
        keep_last = self.max_history_per_stage // 2
        to_compress = history[1:-keep_last]

        summary_parts = []
        for msg in to_compress:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            # Truncate each message in the summary
            summary_parts.append(f"[{role}]: {content[:200]}...")

        compressed = {
            "role": "system",
            "content": f"[Compressed history — {len(to_compress)} messages]\n" +
                       "\n".join(summary_parts[:10]),
            "timestamp": datetime.now().isoformat(),
        }

        self.conversation_history[stage] = [history[0], compressed] + history[-keep_last:]

    def _state_summary(self) -> str:
        """Generate a concise summary of the current research state."""
        lines = [
            "## Research State Summary",
            f"- Phase: {self.state.current_phase}",
            f"- Current stage: {self.state.current_stage}",
            f"- Stages completed: {', '.join(self.state.stages_completed) if self.state.stages_completed else 'none'}",
            f"- Papers found: {len(self.state.literature)}",
            f"- Hypotheses: {len(self.state.hypotheses)}",
            f"- Experiments run: {len(self.state.experiments)}",
            f"- Key findings: {len(self.state.key_findings)}",
        ]

        if self.state.hypotheses:
            lines.append("\n### Hypotheses")
            for h in self.state.hypotheses:
                lines.append(f"  - [{h.status}] {h.statement[:100]}")

        if self.state.methodology:
            lines.append(f"\n### Methodology\n{self.state.methodology[:500]}")

        return "\n".join(lines)

    def _literature_summary(self) -> str:
        """Summarize the literature found so far."""
        lines = ["## Literature Summary"]
        for p in self.state.literature[:15]:  # top 15
            authors = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                authors += " et al."
            lines.append(f"- **{p.title}** ({authors}, {p.year})")
            if p.key_findings:
                lines.append(f"  Key findings: {p.key_findings[:150]}")
        if len(self.state.literature) > 15:
            lines.append(f"  ... and {len(self.state.literature) - 15} more papers")
        return "\n".join(lines)

    def _experiments_summary(self) -> str:
        """Summarize experiment results."""
        lines = ["## Experiment Results"]
        for e in self.state.experiments:
            status_icon = {"completed": "✓", "failed": "✗", "running": "⟳"}.get(e.status, "?")
            lines.append(f"- [{status_icon}] {e.description[:100]}")
            if e.metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in list(e.metrics.items())[:5])
                lines.append(f"  Metrics: {metrics_str}")
        return "\n".join(lines)

    def _reviews_summary(self) -> str:
        """Summarize reviews."""
        lines = ["## Review Summary"]
        for r in self.state.reviews[-3:]:  # last 3 reviews
            score = r.get("score", "?")
            verdict = r.get("verdict", "?")
            lines.append(f"- Score: {score}, Verdict: {verdict}")
            issues = r.get("issues", [])
            for issue in issues[:3]:
                lines.append(f"  - {issue}")
        return "\n".join(lines)
