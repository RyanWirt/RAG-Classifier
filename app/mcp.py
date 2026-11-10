from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MCPTool(Protocol):
    name: str

    def call(self, text: str) -> dict[str, str]: ...


class StdioMCPTool:
    """Calls one tool on an MCP server it launches as a local subprocess."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        tool_name: str,
        argument_name: str = "text",
    ) -> None:
        self.name = name
        self.tool_name = tool_name
        self.argument_name = argument_name
        # cwd is the project root so `-m app...` resolves; env is the SDK's safe default.
        self._params = StdioServerParameters(
            command=command, args=args, cwd=PROJECT_ROOT
        )

    def call(self, text: str) -> dict[str, str]:
        return asyncio.run(self._call(text))

    async def _call(self, text: str) -> dict[str, str]:
        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    self.tool_name, {self.argument_name: text}
                )
        if result.isError:
            raise RuntimeError(f"MCP tool {self.tool_name} returned an error")
        if result.structuredContent is not None:
            return dict(result.structuredContent)
        return json.loads(result.content[0].text)


def terminology_tool() -> StdioMCPTool:
    return StdioMCPTool(
        name="terminology_lookup",
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        tool_name="lookup_terms",
    )
