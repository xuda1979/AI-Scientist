"""Core configuration and utilities for the sciresearch workflow."""
from .config import WorkflowConfig, setup_workflow_logging, timeout_input

from .openai_connection import (
    OpenAIConnectionError,
    OpenAIConnectionManager,
    get_shared_connection_manager,
)

__all__ = [
    'WorkflowConfig',
    'setup_workflow_logging',
    'timeout_input',
    'OpenAIConnectionManager',
    'OpenAIConnectionError',
    'get_shared_connection_manager',
]
