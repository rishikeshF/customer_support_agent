"""Round-robin team graphs used by the urgent path.

Each team is a small graph: `pick_agent` works out whose turn it is, then a
conditional edge sends the ticket to that agent. The counter itself lives in
the caller's state (see `rr_index` in `agentic.workflow`), so the rotation
survives between tickets instead of resetting to the first agent every time.
"""

from typing import Dict, List

from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph

from agentic.agents.experts import agent_swarm_map


class RoundRobinState(MessagesState):
    """State for a team graph: which agents exist and whose turn it is."""

    agent_names: List[str]
    current_agent_index: int


def pick_agent(state: RoundRobinState) -> dict:
    """Normalize the incoming counter into a valid position in the rotation."""
    index = state.get("current_agent_index", 0) % len(state["agent_names"])
    return {"current_agent_index": index}


def route_round_robin(state: RoundRobinState) -> str:
    """Send the ticket to whichever agent's turn it is."""
    return state["agent_names"][state["current_agent_index"]]


def create_team(name: str, agent_pool: List[CompiledStateGraph]) -> CompiledStateGraph:
    workflow = StateGraph(RoundRobinState)
    workflow.add_node("pick_agent", pick_agent)
    for agent in agent_pool:
        workflow.add_node(agent.name, agent)

    workflow.add_edge(START, "pick_agent")
    workflow.add_conditional_edges(
        source="pick_agent",
        path=route_round_robin,
        path_map=[agent.name for agent in agent_pool],
    )
    # No checkpointer here: the team runs inside a node of the main graph, and
    # the rotation counter is carried in the main graph's state.
    return workflow.compile(name=name)


def build_teams(swarm: Dict[str, List[CompiledStateGraph]]) -> Dict[str, CompiledStateGraph]:
    """One team graph per domain, wrapping that domain's pool of agents."""
    return {team_name: create_team(team_name, pool) for team_name, pool in swarm.items()}


agent_teams = build_teams(agent_swarm_map)
