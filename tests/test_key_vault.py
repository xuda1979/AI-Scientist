import json
from cryptography.fernet import Fernet
import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.key_vault import LocalApiKeyVault
from core.openai_connection import OpenAIConnectionError, OpenAIConnectionManager


class DummyOpenAIModule:
    """Simple stand-in for the ``openai`` SDK used in tests."""

    class OpenAI:  # type: ignore[too-few-public-methods]
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        class responses:  # type: ignore[too-few-public-methods]
            @staticmethod
            def with_options(timeout: int):  # pragma: no cover - compatibility no-op
                return DummyOpenAIModule.OpenAI.responses

            @staticmethod
            def create(*args, **kwargs):  # pragma: no cover - unused in tests
                raise NotImplementedError


@pytest.fixture
def dummy_openai(monkeypatch):
    module = DummyOpenAIModule()
    monkeypatch.setitem(__import__('sys').modules, "openai", module)
    return module


def test_local_api_key_vault_roundtrip(tmp_path):
    master_key = Fernet.generate_key()
    vault = LocalApiKeyVault(base_path=tmp_path, master_key=master_key)

    record = vault.store_key("alice", "sk-test-1234")
    assert record.last4 == "1234"
    assert record.status == "valid"

    stored_key = vault.retrieve_key("alice")
    assert stored_key == "sk-test-1234"

    vault.mark_used("alice")
    updated = vault.get_record("alice")
    assert updated is not None and updated.last_used is not None

    vault.delete_key("alice")
    assert vault.retrieve_key("alice") is None


def test_openai_connection_manager_validates_and_tracks_usage(tmp_path, dummy_openai):
    calls = {"count": 0}

    def validator(api_key: str):
        calls["count"] += 1
        if api_key == "bad-key":
            raise OpenAIConnectionError("Rejected test key")
        return {"model_count": "42"}

    vault = LocalApiKeyVault(base_path=tmp_path, master_key=Fernet.generate_key())
    manager = OpenAIConnectionManager(vault=vault, validator=validator)

    status = manager.get_status()
    assert status["status"] == "disconnected"

    metadata = manager.connect("sk-live-7890")
    assert metadata["model_count"] == "42"
    assert calls["count"] == 1

    client = manager.get_client()
    assert isinstance(client, DummyOpenAIModule.OpenAI)
    assert client.api_key == "sk-live-7890"

    refreshed = manager.get_status()
    assert refreshed["status"] == "valid"
    assert refreshed["last4"] == "7890"

    manager.mark_invalid()
    assert manager.get_status()["status"] == "invalid"

    manager.mark_valid()
    assert manager.get_status()["status"] == "valid"

    manager.log_usage("chat.completions", tokens_used=123)
    audit_path = tmp_path / "audit.log"
    assert audit_path.exists()
    entries = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    assert entries[-1]["tokens_used"] == 123

    manager.disconnect()
    assert manager.get_status()["status"] == "disconnected"

    with pytest.raises(OpenAIConnectionError):
        manager.connect("bad-key")
