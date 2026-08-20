"""Optional MCP layer: support operations served over the Model Context Protocol.

The system runs read-only by default. Calling `enable_mcp()` starts the FastMCP
server in `server.py` and rebuilds the expert agents with its operation tools
attached, so they can issue refunds, cancel reservations and change plans.

    from agentic.mcp import enable_mcp
    tools = enable_mcp()

This is opt-in on purpose. `index.ipynb` runs without it; `index_mcp.ipynb`
turns it on. Nothing else in the codebase imports this package, so a broken MCP
setup cannot affect the default path.
"""

from typing import List

from langchain_core.tools import BaseTool

from agentic.agents import experts, agent_swarm_map, agent_teams
from agentic.agents.experts import build_agent_swarm, build_experts
from agentic.agents.teams import build_teams
from agentic.mcp.client import load_mcp_tools

_enabled: List[BaseTool] = []


def enable_mcp() -> List[BaseTool]:
    """
    Attach the MCP operation tools to every expert and team agent.

    The agent registries are updated in place rather than replaced, because the
    workflow looks agents up by name at call time — so the running graph picks
    the new agents up without being rebuilt.

    Returns the tools that were attached, or an empty list if the server could
    not be reached, in which case the system carries on read-only.
    """
    global _enabled

    if _enabled:
        return _enabled

    tools = load_mcp_tools()
    if not tools:
        return []

    experts.update(build_experts(tools))
    agent_swarm_map.update(build_agent_swarm(tools))
    agent_teams.update(build_teams(agent_swarm_map))

    _enabled = tools
    return tools


def mcp_enabled() -> bool:
    """Whether the operation tools are currently attached."""
    return bool(_enabled)


__all__ = ["enable_mcp", "mcp_enabled", "load_mcp_tools"]
