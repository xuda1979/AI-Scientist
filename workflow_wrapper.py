"""
Workflow Wrapper Module
Provides a unified interface for running workflows from both CLI and GUI.
This ensures GUI automatically inherits all CLI functionality and changes.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import threading

from core.config import WorkflowConfig
from sciresearch_workflow import run_workflow, test_time_compute_scaling, DEFAULT_MODEL


class WorkflowParameters:
    """
    Unified parameter container for workflow execution.
    Ensures CLI and GUI use identical parameters.
    """
    
    def __init__(self, **kwargs):
        """Initialize parameters from keyword arguments."""
        # Store all parameters
        self._params = kwargs
        
        # Validate required parameters
        if not self._params.get("output_dir"):
            raise ValueError("output_dir is required")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for logging/serialization."""
        return self._params.copy()
    
    @classmethod
    def from_gui(cls, gui_params: Dict[str, Any]) -> "WorkflowParameters":
        """
        Create WorkflowParameters from GUI parameter dictionary.
        This method handles GUI-specific transformations.
        """
        # Convert GUI parameters to workflow parameters
        params = gui_params.copy()
        
        # Handle boolean inversions (GUI uses 'disable_' but workflow uses 'enable_')
        if "disable_blueprint_planning" in params:
            params["enable_blueprint_planning"] = not bool(params.pop("disable_blueprint_planning", False))
        
        return cls(**params)
    
    @classmethod
    def from_cli(cls, cli_args) -> "WorkflowParameters":
        """
        Create WorkflowParameters from CLI argument namespace.
        This method handles CLI-specific transformations.
        """
        # Convert CLI args to dictionary
        params = vars(cli_args).copy()
        
        # Handle disable_blueprint_planning if it exists
        if "disable_blueprint_planning" in params:
            params["enable_blueprint_planning"] = not bool(params.pop("disable_blueprint_planning", False))
        
        return cls(**params)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value with optional default."""
        return self._params.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get parameter value."""
        return self._params[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self._params


def prepare_workflow_config(params: WorkflowParameters) -> WorkflowConfig:
    """
    Prepare WorkflowConfig from parameters.
    This centralizes all config preparation logic.
    """
    config_path = params.get("config_path")
    
    if config_path:
        config = WorkflowConfig.from_file(Path(config_path))
    else:
        config = WorkflowConfig()
    
    # Apply parameter overrides to config
    # This mapping is the SINGLE SOURCE OF TRUTH for CLI/GUI → Config translation
    
    # PDF review settings
    if params.get("enable_pdf_review") is not None and params.get("disable_pdf_review") is not None:
        config.enable_pdf_review = bool(params["enable_pdf_review"]) and not bool(params["disable_pdf_review"])
    elif params.get("enable_pdf_review") is not None:
        config.enable_pdf_review = bool(params["enable_pdf_review"])
    elif params.get("disable_pdf_review") is not None:
        config.enable_pdf_review = not bool(params["disable_pdf_review"])
    
    # Reference and figure validation
    if params.get("check_references") is not None and params.get("skip_reference_check") is not None:
        config.reference_validation = bool(params["check_references"]) and not bool(params["skip_reference_check"])
    elif params.get("check_references") is not None:
        config.reference_validation = bool(params["check_references"])
    elif params.get("skip_reference_check") is not None:
        config.reference_validation = not bool(params["skip_reference_check"])
    
    if params.get("validate_figures") is not None and params.get("skip_figure_validation") is not None:
        config.figure_validation = bool(params["validate_figures"]) and not bool(params["skip_figure_validation"])
    elif params.get("validate_figures") is not None:
        config.figure_validation = bool(params["validate_figures"])
    elif params.get("skip_figure_validation") is not None:
        config.figure_validation = not bool(params["skip_figure_validation"])
    
    # Ideation settings
    if params.get("enable_ideation") is not None and params.get("skip_ideation") is not None:
        config.research_ideation = bool(params["enable_ideation"]) and not bool(params["skip_ideation"])
    elif params.get("enable_ideation") is not None:
        config.research_ideation = bool(params["enable_ideation"])
    elif params.get("skip_ideation") is not None:
        config.research_ideation = not bool(params["skip_ideation"])
    
    # Diff tracking
    if params.get("output_diffs") is not None and params.get("no_output_diffs") is not None:
        config.diff_output_tracking = bool(params["output_diffs"]) and not bool(params["no_output_diffs"])
    elif params.get("output_diffs") is not None:
        config.diff_output_tracking = bool(params["output_diffs"])
    elif params.get("no_output_diffs") is not None:
        config.diff_output_tracking = not bool(params["no_output_diffs"])
    
    # Content protection
    if params.get("disable_content_protection") is not None:
        config.content_protection = not bool(params["disable_content_protection"])
    
    if params.get("auto_approve_changes") is not None:
        config.auto_approve_changes = bool(params["auto_approve_changes"])
    
    if params.get("content_protection_threshold") is not None:
        config.content_protection_threshold = float(params["content_protection_threshold"])
    
    # Workflow execution settings
    if params.get("no_early_stopping") is not None:
        config.no_early_stopping = bool(params["no_early_stopping"])
    
    if params.get("use_test_time_scaling") is not None:
        config.use_test_time_scaling = bool(params["use_test_time_scaling"])
    
    if params.get("revision_candidates") is not None:
        config.revision_candidates = int(params["revision_candidates"])
    
    if params.get("initial_draft_candidates") is not None:
        config.initial_draft_candidates = int(params["initial_draft_candidates"])
    elif params.get("draft_candidates") is not None:
        config.initial_draft_candidates = int(params["draft_candidates"])
    
    # Quality and iteration settings
    if params.get("quality_threshold") is not None:
        config.quality_threshold = float(params["quality_threshold"])
    
    if params.get("max_iterations") is not None:
        config.max_iterations = int(params["max_iterations"])
    
    if params.get("request_timeout") is not None:
        config.request_timeout = int(params["request_timeout"])
    
    if params.get("max_retries") is not None:
        config.max_retries = int(params["max_retries"])
    
    return config


def execute_workflow(params: WorkflowParameters, cancel_event: Optional[threading.Event] = None) -> Path:
    """
    Execute the main workflow.
    This is the SINGLE ENTRY POINT for both CLI and GUI.
    
    Any changes to workflow execution should be made HERE, and both
    CLI and GUI will automatically inherit the changes.
    """
    # Prepare configuration
    config = prepare_workflow_config(params)
    
    # Save config if requested
    save_path = params.get("save_config_path")
    if save_path:
        config.save_to_file(Path(save_path))
        print(f"Configuration saved to {save_path}")
        # If only saving config, return early
        if params.get("config_only"):
            return Path(save_path).parent
    
    # Extract core parameters
    output_dir = Path(str(params["output_dir"]))
    topic = str(params.get("topic", ""))
    field = str(params.get("field", ""))
    question = str(params.get("question", ""))
    model = str(params.get("model", DEFAULT_MODEL))
    
    # Execute workflow
    result_dir = run_workflow(
        topic=topic,
        field=field,
        question=question,
        output_dir=output_dir,
        model=model,
        request_timeout=(None if params.get("request_timeout") == 0 else params.get("request_timeout")),
        max_retries=int(params.get("max_retries", 3)),
        max_iterations=int(params.get("max_iterations", 4)),
        modify_existing=bool(params.get("modify_existing", False)),
        strict_singletons=bool(params.get("strict_singletons", True)),
        python_exec=params.get("python_exec"),
        quality_threshold=float(params.get("quality_threshold", 1.0)),
        check_references=config.reference_validation,
        validate_figures=config.figure_validation,
        user_prompt=params.get("user_prompt"),
        config=config,
        enable_ideation=config.research_ideation,
        specify_idea=params.get("specify_idea"),
        num_ideas=int(params.get("num_ideas", 15)),
        output_diffs=config.diff_output_tracking,
        document_type=str(params.get("document_type", "auto")),
        enable_blueprint_planning=params.get("enable_blueprint_planning"),
        cancel_event=cancel_event,
    )
    
    return result_dir


def execute_test_scaling(params: WorkflowParameters) -> bool:
    """
    Execute test-time compute scaling analysis.
    This is the SINGLE ENTRY POINT for scaling tests from both CLI and GUI.
    """
    # Parse scaling candidates
    candidate_text = params.get("scaling_candidates", "3,5,7,10")
    try:
        candidates = [int(value.strip()) for value in str(candidate_text).split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid scaling candidates: {candidate_text}") from exc
    
    if not candidates:
        raise ValueError("At least one scaling candidate is required when test scaling is enabled.")
    
    model = str(params.get("model", DEFAULT_MODEL))
    scaling_prompt = params.get("scaling_prompt")
    scaling_timeout = int(params.get("scaling_timeout", 1800))
    
    print("Running test-time compute scaling analysis...")
    result = test_time_compute_scaling(
        model=model,
        test_prompt=scaling_prompt,
        candidate_counts=candidates,
        timeout_base=scaling_timeout,
    )
    
    if result:
        print("Test-time compute scaling completed successfully.")
        return True
    else:
        print("Test-time compute scaling failed.")
        return False


def execute(params: WorkflowParameters, cancel_event: Optional[threading.Event] = None) -> Path:
    """
    Universal execution entry point.
    Automatically routes to workflow or scaling test based on parameters.
    
    This is the ONLY function that CLI and GUI should call.
    """
    # Check if this is a test scaling run
    if params.get("test_scaling"):
        success = execute_test_scaling(params)
        if not success:
            raise RuntimeError("Test-time compute scaling failed")
        # Return a dummy path for scaling tests
        return Path(".")
    
    # Otherwise, run normal workflow
    return execute_workflow(params, cancel_event)


# Convenience function for backward compatibility
def run_from_gui(gui_params: Dict[str, Any], cancel_event: Optional[threading.Event] = None) -> Path:
    """
    Execute workflow from GUI parameters.
    This ensures GUI automatically inherits all workflow changes.
    """
    params = WorkflowParameters.from_gui(gui_params)
    return execute(params, cancel_event)


def run_from_cli(cli_args) -> Path:
    """
    Execute workflow from CLI arguments.
    This ensures CLI uses the same execution path as GUI.
    """
    params = WorkflowParameters.from_cli(cli_args)
    return execute(params)
