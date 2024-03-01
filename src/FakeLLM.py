"""Fake LLM wrapper for testing purposes."""
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
class FakeStaticLLM(LLM):
    """Fake Static LLM wrapper for testing purposes."""

    response:str

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-static"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Return static response."""
        return self.response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}
    

class FakePromptCopyLLM(LLM):
    """Fake Prompt Copy LLM wrapper for testing purposes."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-prompt-copy"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Return prompt."""
        return prompt

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

class FakeListLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    responses: List
    i: int = 0

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        response = self.responses[self.i]
        self.i += 1
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}