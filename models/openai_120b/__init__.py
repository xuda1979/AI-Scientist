class OpenAI120B:
    """Placeholder implementation of the OpenAI OSS 120B model."""
    def __init__(self):
        self.device = "cpu"
    def to(self, device: str):
        self.device = device
        return self
    def __call__(self, prompt: str) -> str:
        return f"[openai-oss-120b placeholder on {self.device}]"

__all__ = ["OpenAI120B"]
