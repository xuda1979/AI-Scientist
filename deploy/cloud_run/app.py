"""FastAPI application for running the AI Scientist workflow on Cloud Run.

This service exposes a minimal REST API that allows clients to trigger
AI Scientist workflow runs and poll for their completion status.  Results can
optionally be archived to a Google Cloud Storage bucket when the
``RESULTS_GCS_BUCKET`` environment variable is provided.
"""
from __future__ import annotations

import os
import threading
import uuid
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sciresearch_workflow import WorkflowCancelled, run_workflow
from core.config import WorkflowConfig

app = FastAPI(
    title="AI Scientist Workflow Service",
    description=(
        "Trigger AI Scientist research workflow runs and poll their status. "
        "Each run executes the full pipeline and stores outputs inside the "
        "container file system or an optional Cloud Storage bucket."
    ),
    version="1.0.0",
)

# Thread pool used to run workflows asynchronously so API requests return quickly.
_MAX_WORKERS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

# In-memory registry that tracks the lifecycle of workflow executions.
_JOB_REGISTRY: Dict[str, Dict[str, Any]] = {}
_REGISTRY_LOCK = threading.Lock()


class WorkflowRequest(BaseModel):
    """Payload for creating a new workflow run."""

    topic: str = Field(..., description="High-level research topic")
    field: str = Field(..., description="Scientific or engineering field")
    question: str = Field(..., description="Specific research question to explore")
    output_subdir: Optional[str] = Field(
        default=None,
        description=(
            "Optional sub-directory inside OUTPUT_BASE_DIR where the run "
            "artifacts will be stored. When omitted a directory derived from "
            "the job identifier is used."
        ),
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model identifier to pass through to the workflow",
    )
    max_iterations: Optional[int] = Field(
        default=4, description="Maximum number of review/revision iterations"
    )
    enable_ideation: Optional[bool] = Field(
        default=True, description="Whether to run the ideation stage"
    )
    specify_idea: Optional[str] = Field(
        default=None,
        description="Provide a concrete idea instead of auto-ideation",
    )
    num_ideas: Optional[int] = Field(
        default=15, description="Number of candidate ideas to evaluate"
    )
    quality_threshold: Optional[float] = Field(
        default=1.0, description="Quality score required to stop the workflow"
    )
    check_references: Optional[bool] = Field(
        default=True, description="Enable automated reference validation"
    )
    validate_figures: Optional[bool] = Field(
        default=True, description="Enable LaTeX figure/table validation"
    )
    user_prompt: Optional[str] = Field(
        default="",
        description=(
            "Additional global instructions injected into every prompt.  "
            "Set to an empty string to skip custom instructions."
        ),
    )
    document_type: Optional[str] = Field(
        default="auto", description="Explicit document type override"
    )
    enable_blueprint_planning: Optional[bool] = Field(
        default=None,
        description="Override blueprint planning toggle (otherwise config default is used)",
    )


class WorkflowStatusResponse(BaseModel):
    """Serialized representation of a workflow job state."""

    job_id: str
    status: str
    submitted_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    result_dir: Optional[str]
    artifact_uri: Optional[str]
    error: Optional[str]


def _serialize_status(job_id: str, record: Dict[str, Any]) -> WorkflowStatusResponse:
    """Convert the internal job dictionary into a response model."""

    return WorkflowStatusResponse(
        job_id=job_id,
        status=record["status"],
        submitted_at=record["submitted_at"],
        started_at=record.get("started_at"),
        finished_at=record.get("finished_at"),
        result_dir=record.get("result_dir"),
        artifact_uri=record.get("artifact_uri"),
        error=record.get("error"),
    )


def _archive_to_gcs(job_id: str, result_dir: Path) -> Optional[str]:
    """Archive the workflow outputs to a Cloud Storage bucket if configured."""

    bucket_name = os.getenv("RESULTS_GCS_BUCKET")
    if not bucket_name:
        return None

    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "RESULTS_GCS_BUCKET is set but google-cloud-storage is not installed"
        ) from exc

    prefix = os.getenv("RESULTS_GCS_PREFIX", "ai-scientist")
    archive_name = f"{prefix.rstrip('/')}/{job_id}.zip"

    temp_dir = Path(tempfile.gettempdir())
    archive_path = temp_dir / f"{job_id}.zip"

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in result_dir.rglob("*"):
            if file_path.is_file():
                zip_file.write(file_path, file_path.relative_to(result_dir))

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(archive_name)
    blob.upload_from_filename(str(archive_path))

    try:
        archive_path.unlink()
    except OSError:
        pass

    return f"gs://{bucket_name}/{archive_name}"


def _run_workflow_async(job_id: str, payload: WorkflowRequest) -> None:
    """Execute the workflow and update the registry with progress information."""

    with _REGISTRY_LOCK:
        _JOB_REGISTRY[job_id]["status"] = "running"
        _JOB_REGISTRY[job_id]["started_at"] = datetime.utcnow()

    output_base = Path(os.getenv("OUTPUT_BASE_DIR", "/workspace/output"))
    output_base.mkdir(parents=True, exist_ok=True)

    subdir = payload.output_subdir.strip() if payload.output_subdir else job_id
    safe_subdir = Path(subdir).name  # Prevent directory traversal
    run_output_dir = output_base / safe_subdir

    try:
        config = WorkflowConfig()
        result_path = run_workflow(
            topic=payload.topic,
            field=payload.field,
            question=payload.question,
            output_dir=run_output_dir,
            model=payload.model or os.getenv("SCI_MODEL", "gpt-5"),
            max_iterations=
            payload.max_iterations if payload.max_iterations is not None else config.max_iterations,
            quality_threshold=
            payload.quality_threshold if payload.quality_threshold is not None else config.quality_threshold,
            check_references=payload.check_references
            if payload.check_references is not None
            else True,
            validate_figures=payload.validate_figures
            if payload.validate_figures is not None
            else True,
            user_prompt=payload.user_prompt if payload.user_prompt is not None else "",
            config=config,
            enable_ideation=payload.enable_ideation
            if payload.enable_ideation is not None
            else True,
            specify_idea=payload.specify_idea,
            num_ideas=payload.num_ideas if payload.num_ideas is not None else 15,
            output_diffs=False,
            document_type=payload.document_type if payload.document_type is not None else "auto",
            enable_blueprint_planning=payload.enable_blueprint_planning,
        )

        artifact_uri = _archive_to_gcs(job_id, result_path)

        with _REGISTRY_LOCK:
            _JOB_REGISTRY[job_id]["status"] = "succeeded"
            _JOB_REGISTRY[job_id]["finished_at"] = datetime.utcnow()
            _JOB_REGISTRY[job_id]["result_dir"] = str(result_path)
            if artifact_uri:
                _JOB_REGISTRY[job_id]["artifact_uri"] = artifact_uri

    except WorkflowCancelled as exc:
        with _REGISTRY_LOCK:
            _JOB_REGISTRY[job_id]["status"] = "cancelled"
            _JOB_REGISTRY[job_id]["finished_at"] = datetime.utcnow()
            _JOB_REGISTRY[job_id]["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive broad catch for API surface
        with _REGISTRY_LOCK:
            _JOB_REGISTRY[job_id]["status"] = "failed"
            _JOB_REGISTRY[job_id]["finished_at"] = datetime.utcnow()
            _JOB_REGISTRY[job_id]["error"] = str(exc)


@app.post("/jobs", response_model=WorkflowStatusResponse, status_code=202)
def create_job(payload: WorkflowRequest) -> WorkflowStatusResponse:
    """Schedule a workflow run and return its identifier."""

    job_id = uuid.uuid4().hex
    record = {
        "status": "queued",
        "submitted_at": datetime.utcnow(),
        "result_dir": None,
        "artifact_uri": None,
        "error": None,
    }

    with _REGISTRY_LOCK:
        _JOB_REGISTRY[job_id] = record

    _EXECUTOR.submit(_run_workflow_async, job_id, payload)

    return _serialize_status(job_id, record)


@app.get("/jobs/{job_id}", response_model=WorkflowStatusResponse)
def get_job_status(job_id: str) -> WorkflowStatusResponse:
    """Retrieve the status of a previously scheduled workflow."""

    with _REGISTRY_LOCK:
        record = _JOB_REGISTRY.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return _serialize_status(job_id, record)


@app.get("/healthz")
def health_check() -> Dict[str, str]:
    """Lightweight endpoint used by Cloud Run for container health checks."""

    return {"status": "ok"}
