"""Secure storage for user-provided API keys."""
from __future__ import annotations

import json
import os
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Optional

from cryptography.fernet import Fernet


__all__ = ["ApiKeyVaultError", "ApiKeyRecord", "LocalApiKeyVault"]


@dataclass
class ApiKeyRecord:
    """Metadata for a stored API key."""

    user_id: str
    key_hash: str
    last4: str
    created_at: str
    updated_at: str
    last_used: Optional[str]
    status: str
    encrypted_key: str

    def redacted(self) -> Dict[str, Optional[str]]:
        """Return metadata safe for UI display."""
        return {
            "user_id": self.user_id,
            "key_hash": self.key_hash,
            "last4": self.last4,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used": self.last_used,
            "status": self.status,
        }


class ApiKeyVaultError(RuntimeError):
    """Raised when the vault cannot service a request."""


class LocalApiKeyVault:
    """File-based vault that encrypts API keys at rest using Fernet."""

    DEFAULT_DIRNAME = ".sciresearch"
    DEFAULT_FILENAME = "vault.json"
    MASTER_KEY_FILENAME = "vault.key"

    def __init__(
        self,
        base_path: Optional[Path] = None,
        *,
        master_key: Optional[bytes] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._base_path = base_path or Path.home() / self.DEFAULT_DIRNAME
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._vault_path = self._base_path / self.DEFAULT_FILENAME
        self._master_key_path = self._base_path / self.MASTER_KEY_FILENAME

        if master_key is None:
            master_key = self._load_or_create_master_key()
        self._fernet = Fernet(master_key)
        self._state: Dict[str, Dict[str, str]] = {}
        self._last_loaded_mtime: float = 0.0
        with self._lock:
            self._load_state_locked()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_key(self, user_id: str, api_key: str, *, status: str = "valid") -> ApiKeyRecord:
        """Encrypt and persist an API key for the provided user."""
        if not api_key:
            raise ApiKeyVaultError("API key cannot be empty")

        now = self._now()
        key_hash = self._hash_key(api_key)
        last4 = api_key[-4:] if len(api_key) >= 4 else api_key
        encrypted_key = self._fernet.encrypt(api_key.encode("utf-8")).decode("utf-8")

        with self._lock:
            self._refresh_state_locked()
            record = self._state.get(user_id)
            created_at = record.get("created_at") if record else now
            record_data = {
                "user_id": user_id,
                "key_hash": key_hash,
                "last4": last4,
                "created_at": created_at,
                "updated_at": now,
                "last_used": record.get("last_used") if record else None,
                "status": status,
                "encrypted_key": encrypted_key,
            }
            self._state[user_id] = record_data
            self._persist_state()
        return ApiKeyRecord(**record_data)

    def retrieve_key(self, user_id: str) -> Optional[str]:
        """Return the decrypted API key for the user."""
        with self._lock:
            self._refresh_state_locked()
            record = self._state.get(user_id)
            if not record:
                return None
            encrypted_key = record.get("encrypted_key")
            if not encrypted_key:
                return None
            try:
                decrypted = self._fernet.decrypt(encrypted_key.encode("utf-8"))
            except Exception as exc:  # pragma: no cover - cryptography sanity check
                raise ApiKeyVaultError("Failed to decrypt API key") from exc
        return decrypted.decode("utf-8")

    def delete_key(self, user_id: str) -> None:
        """Remove a stored key and metadata for the user."""
        with self._lock:
            self._refresh_state_locked()
            if user_id in self._state:
                del self._state[user_id]
                self._persist_state()

    def get_record(self, user_id: str) -> Optional[ApiKeyRecord]:
        """Return the full record for a user (including encrypted key)."""
        with self._lock:
            self._refresh_state_locked()
            record = self._state.get(user_id)
            if not record:
                return None
            return ApiKeyRecord(**record)

    def update_status(self, user_id: str, status: str) -> Optional[ApiKeyRecord]:
        """Update the stored status for a user and return the new record."""
        with self._lock:
            self._refresh_state_locked()
            record = self._state.get(user_id)
            if not record:
                return None
            record["status"] = status
            record["updated_at"] = self._now()
            self._persist_state()
            return ApiKeyRecord(**record)

    def mark_used(self, user_id: str) -> Optional[ApiKeyRecord]:
        """Update the last-used timestamp for the stored key."""
        with self._lock:
            self._refresh_state_locked()
            record = self._state.get(user_id)
            if not record:
                return None
            record["last_used"] = self._now()
            record["updated_at"] = record["last_used"]
            self._persist_state()
            return ApiKeyRecord(**record)

    def audit_event(self, user_id: str, *, endpoint: str, tokens_used: Optional[int] = None) -> None:
        """Append an audit entry for API usage."""
        audit_path = self._base_path / "audit.log"
        entry = {
            "timestamp": self._now(),
            "user_id": user_id,
            "endpoint": endpoint,
            "tokens_used": tokens_used,
        }
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + os.linesep)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_state_locked(self) -> None:
        if not self._vault_path.exists():
            self._state = {}
            self._last_loaded_mtime = 0.0
            return
        try:
            with self._vault_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ApiKeyVaultError("Vault file is corrupted") from exc
        records = payload.get("records", {})
        self._state = {user_id: data for user_id, data in records.items()}
        self._last_loaded_mtime = self._get_vault_mtime()

    def _persist_state(self) -> None:
        payload = {"records": self._state}
        tmp_path = self._vault_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, self._vault_path)
        self._last_loaded_mtime = self._get_vault_mtime()

    def _refresh_state_locked(self) -> None:
        if not self._vault_path.exists():
            if self._state:
                self._state = {}
            self._last_loaded_mtime = 0.0
            return

        current_mtime = self._get_vault_mtime()
        if current_mtime > self._last_loaded_mtime:
            self._load_state_locked()

    def _get_vault_mtime(self) -> float:
        try:
            return self._vault_path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def _load_or_create_master_key(self) -> bytes:
        if self._master_key_path.exists():
            key_bytes = self._master_key_path.read_bytes()
            return key_bytes.strip()

        key_bytes = Fernet.generate_key()
        with self._master_key_path.open("wb") as handle:
            handle.write(key_bytes)
        os.chmod(self._master_key_path, stat.S_IRUSR | stat.S_IWUSR)
        return key_bytes

    @staticmethod
    def _hash_key(api_key: str) -> str:
        import hashlib

        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
