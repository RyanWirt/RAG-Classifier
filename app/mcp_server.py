"""Terminology MCP server over stdio. Run with: python -m app.mcp_server"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

TERMS = {
    "refund": "A payment returned to a customer.",
    "personal data": "Information relating to an identified person.",
    "service outage": "An interruption that makes a service unavailable.",
}

server = FastMCP("terminology", log_level="WARNING")


@server.tool()
def lookup_terms(text: str) -> dict[str, str]:
    """Return definitions for known terms that appear in the text."""
    lowered = text.casefold()
    return {term: meaning for term, meaning in TERMS.items() if term in lowered}


if __name__ == "__main__":
    server.run(transport="stdio")
