"""Load the MCP support-operation tools so LangGraph agents can call them.

The server in `server.py` runs as a child process over stdio. This module
starts it, asks it what tools it has, and hands back LangChain tools.

Loading is deliberately forgiving: if the server cannot start, the support
system still works with its read-only tools rather than failing to import.
"""

import asyncio
import sys
from typing import List

from langchain_core.tools import BaseTool

from agentic.config import BASE_DIR
from agentic.observability import log_event

SERVER_CONNECTION = {
    "cultpass_operations": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "agentic.mcp.server"],
        "cwd": str(BASE_DIR),
    }
}


async def _fetch_tools() -> List[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(SERVER_CONNECTION)
    return await client.get_tools()


def load_mcp_tools() -> List[BaseTool]:
    """
    Start the MCP server and return its tools as LangChain tools.

    Returns an empty list if the server is unavailable, so a broken MCP setup
    degrades the system to read-only rather than breaking it outright.
    """
    try:
        tools = _run_async(_fetch_tools())
        log_event("mcp_tools_loaded", tools=[t.name for t in tools])
        return tools
    except Exception as error:
        log_event("mcp_unavailable", error=str(error))
        print(f"MCP tools unavailable, continuing without them: {error}", file=sys.stderr)
        return []


def _run_async(coro):
    """
    Run a coroutine from sync code, including inside a running notebook loop.

    A notebook kernel already has a loop running, so asyncio.run() would raise.
    In that case we hand the work to a separate thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
