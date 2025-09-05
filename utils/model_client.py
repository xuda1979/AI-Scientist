from dataclasses import dataclass
from typing import List, Dict, Any
import requests


@dataclass
class OSS120BClient:
    """Simple client for interacting with an external OSS 120B model server."""
    base_url: str
    api_key: str

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def ping(self) -> bool:
        """Check if the server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/health", headers=self._headers(), timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def chat(self, messages: List[Dict[str, str]], timeout: int | None = None) -> str:
        """Send a chat completion request to the OSS 120B server."""
        payload = {"messages": messages}
        headers = self._headers()
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout or 60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"OSS120B request failed: {e}")
