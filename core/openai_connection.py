"""Utilities for managing user-supplied OpenAI API keys."""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from .key_vault import ApiKeyVaultError, LocalApiKeyVault


__all__ = [
    "OpenAIConnectionError",
    "OpenAIConnectionManager",
    "get_shared_connection_manager",
]


logger = logging.getLogger(__name__)


class OpenAIConnectionError(RuntimeError):
    """Raised when OpenAI connectivity cannot be established."""


Validator = Callable[[str], Dict[str, str]]


class OpenAIConnectionManager:
    """High-level helper that validates and stores OpenAI API keys."""

    DEFAULT_USER_ID = "default"

    def __init__(
        self,
        vault: Optional[LocalApiKeyVault] = None,
        *,
        validator: Optional[Validator] = None,
    ) -> None:
        self._vault = vault or LocalApiKeyVault()
        self._validator = validator or self._default_validator

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self, api_key: str, *, user_id: str = DEFAULT_USER_ID) -> Dict[str, Optional[str]]:
        """Validate and persist the provided OpenAI API key."""
        if not api_key:
            raise OpenAIConnectionError("API key is required")

        try:
            metadata = self._validator(api_key)
        except OpenAIConnectionError:
            raise
        except Exception as exc:  # pragma: no cover - external library failure
            raise OpenAIConnectionError(f"Failed to validate API key: {exc}") from exc

        record = self._vault.store_key(user_id, api_key, status="valid")
        redacted = record.redacted()
        redacted.update(metadata or {})
        return redacted

    def disconnect(self, *, user_id: str = DEFAULT_USER_ID) -> None:
        """Forget a previously stored key."""
        self._vault.delete_key(user_id)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def get_status(self, *, user_id: str = DEFAULT_USER_ID) -> Dict[str, Optional[str]]:
        record = self._vault.get_record(user_id)
        if not record:
            return {"status": "disconnected", "user_id": user_id}
        return record.redacted()

    def is_connected(self, *, user_id: str = DEFAULT_USER_ID) -> bool:
        return self._vault.get_record(user_id) is not None

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------
    def get_client(self, *, user_id: str = DEFAULT_USER_ID):
        """Return a configured ``openai.OpenAI`` client for the stored key."""
        api_key = self._vault.retrieve_key(user_id)
        if not api_key:
            raise OpenAIConnectionError("No OpenAI API key configured. Connect via the GUI settings.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - library missing
            raise OpenAIConnectionError("The 'openai' package is not installed") from exc

        client = OpenAI(api_key=api_key)
        self._vault.mark_used(user_id)
        return client

    def mark_invalid(self, *, user_id: str = DEFAULT_USER_ID) -> None:
        """Flag the stored key as invalid to prompt reconnection."""
        if not self._vault.update_status(user_id, "invalid"):
            logger.warning("Attempted to mark missing key as invalid for user %s", user_id)

    def mark_valid(self, *, user_id: str = DEFAULT_USER_ID) -> None:
        """Mark the key as valid again (used after a successful retry)."""
        if not self._vault.update_status(user_id, "valid"):
            logger.warning("Attempted to mark missing key as valid for user %s", user_id)

    def log_usage(
        self,
        endpoint: str,
        *,
        tokens_used: Optional[int] = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> None:
        """Record an audit event for monitoring and metering."""
        try:
            self._vault.audit_event(user_id, endpoint=endpoint, tokens_used=tokens_used)
        except ApiKeyVaultError as exc:  # pragma: no cover - audit failure is non-fatal
            logger.warning("Failed to persist audit event: %s", exc)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    def _default_validator(self, api_key: str) -> Dict[str, str]:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - library missing
            raise OpenAIConnectionError("The 'openai' package is not installed") from exc

        client = OpenAI(api_key=api_key)
        try:
            models = client.models.list()
        except Exception as exc:
            raise OpenAIConnectionError("The provided OpenAI API key could not be validated") from exc

        # Provide minimal metadata for display purposes
        try:
            model_count = len(models.data) if hasattr(models, "data") else None
        except Exception:  # pragma: no cover - optional metadata extraction
            model_count = None

        metadata: Dict[str, str] = {}
        if model_count is not None:
            metadata["model_count"] = str(model_count)
        return metadata


_SHARED_MANAGER: Optional[OpenAIConnectionManager] = None


def get_shared_connection_manager() -> OpenAIConnectionManager:
    """Return a process-wide :class:`OpenAIConnectionManager` instance."""

    global _SHARED_MANAGER
    if _SHARED_MANAGER is None:
        _SHARED_MANAGER = OpenAIConnectionManager()
    return _SHARED_MANAGER
