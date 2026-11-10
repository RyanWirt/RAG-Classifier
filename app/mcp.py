from __future__ import annotations

from typing import Protocol


class MCPTool(Protocol):
    name: str

    def call(self, text: str) -> dict[str, str]: ...


class MockTerminologyTool:
    """Local example adapter; replace with a client for a real MCP server."""

    name = "mock_terminology_lookup"
    _terms = {
        "refund": "A payment returned to a customer.",
        "personal data": "Information relating to an identified person.",
        "service outage": "An interruption that makes a service unavailable.",
    }

    def call(self, text: str) -> dict[str, str]:
        lowered = text.casefold()
        return {
            term: definition
            for term, definition in self._terms.items()
            if term in lowered
        }
