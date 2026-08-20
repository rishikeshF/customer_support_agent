"""Generator: rebuilds index.ipynb and index_mcp.ipynb from the agentic/ sources.

Keeping the notebooks generated rather than hand-copied means they cannot drift
away from the code that actually runs. Re-run after changing agentic/.

    index.ipynb       the system as it runs by default: read-only tools
    index_mcp.ipynb   the same system with the MCP operation tools attached

The two share every cell up to "Try it"; the MCP notebook adds the section that
starts the server and rebuilds the agents with its tools.
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def source_of(relative_path: str) -> str:
    """Read a module, dropping its module docstring and intra-package imports."""
    text = (BASE / relative_path).read_text(encoding="utf-8")

    # Drop the leading module docstring.
    if text.startswith('"""'):
        text = text[text.index('"""', 3) + 3 :]

    kept, skipping = [], False
    for line in text.splitlines():
        if line.startswith("from agentic"):
            # Intra-package imports are meaningless once flattened into one file.
            skipping = "(" in line and ")" not in line
            continue
        if skipping:
            skipping = ")" not in line
            continue
        kept.append(line)

    return "\n".join(kept).strip()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


AGENTIC_TOOLS = '''agentic_tools = [
    search_rag_knowledge_base,
    get_account_user_by_external_id,
    get_user_subscription,
    get_user_reservations,
    get_ticket_details,
    get_ticket_messages,
    remember_customer_preference,
    recall_customer_preferences,
]

# escalate_ticket is deliberately not in this list: only the escalation agent
# may hand a ticket to a human, so an expert cannot quietly escalate its way
# out of a hard question.'''

DEMO = '''DEMO_CUSTOMER = "a4ab87"  # external_user_id of Alice Kingsley
DEMO_TICKET = "233314ae-815b-46aa-8466-c4163835b224"

run_support_query(
    query="How do refunds work on my CultPass card?",
    customer_id=DEMO_CUSTOMER,
    ticket_id=DEMO_TICKET,
)'''

DEMO_URGENT = '''run_support_query(
    query="My card was charged twice this morning, I need this fixed now!",
    customer_id=DEMO_CUSTOMER,
    ticket_id=DEMO_TICKET,
)'''

DEMO_ESCALATION = '''run_support_query(
    query="I want to talk to a real human agent right now.",
    customer_id=DEMO_CUSTOMER,
    ticket_id=DEMO_TICKET,
)'''

DRAW = '''from IPython.display import Image, display

display(Image(support_graph.get_graph().draw_mermaid_png()))'''

DEMO_UNANSWERABLE = '''# Nothing in the knowledge base covers this, so it escalates
# even though the customer never asked for a person.
run_support_query(
    query="What is the airspeed velocity of an unladen swallow?",
    customer_id=DEMO_CUSTOMER,
    ticket_id=DEMO_TICKET,
)'''

DEMO_LOG = '''# Every decision in the run above, replayed from the log.
for record in read_log(ticket_id=DEMO_TICKET)[-12:]:
    print(record["event"], "->", {k: v for k, v in record.items() if k not in ("ts", "event")})'''

DEMO_MEMORY = '''# Long-term memory: what we know about this customer across sessions.
print("preferences:", load_preferences_text(DEMO_CUSTOMER))
print("past issues:", load_history_text(DEMO_CUSTOMER))

# And the durable conversation record on the ticket itself.
get_ticket_messages.invoke({"ticket_id": DEMO_TICKET})'''

MCP_ENABLE = '''# Start the MCP server (as a child process) and take its tools.
mcp_tools = load_mcp_tools()
print("operations available:", [t.name for t in mcp_tools])'''

MCP_ATTACH = '''# Rebuild the agents with the operation tools attached. The graph looks agents
# up by name at call time, so it picks these up without being recompiled.
experts.update(build_experts(mcp_tools))
agent_swarm_map.update(build_agent_swarm(mcp_tools))
agent_teams.update(build_teams(agent_swarm_map))

sorted(experts["billing"].get_graph().nodes["tools"].data.tools_by_name)'''

MCP_DEMO = '''# The billing expert can now act on the account, not just describe the policy.
run_support_query(
    query="Please cancel my reservation for the workshop and refund it.",
    customer_id=DEMO_CUSTOMER,
    ticket_id=DEMO_TICKET,
)'''


def mcp_cells() -> list:
    """The section that turns the read-only system into an acting one."""
    server_source = (BASE / "agentic/mcp/server.py").read_text(encoding="utf-8")
    return [
        md(
            "## Support operations over MCP\n"
            "\n"
            "Everything above is read-only: the agents look things up and explain. "
            "The write operations — refunds, cancellations, subscription and account "
            "changes — live behind an MCP server instead, so the agents reach them "
            "over a protocol boundary they cannot bypass, and the same server can be "
            "pointed at a real CultPass backend later without touching agent code.\n"
            "\n"
            "| Operation | What it does |\n"
            "| --- | --- |\n"
            "| `process_refund` | Refunds a reservation and records the refund |\n"
            "| `cancel_reservation` | Cancels a booking and releases the slot |\n"
            "| `set_subscription_status` | Pauses, resumes or cancels a subscription |\n"
            "| `change_subscription_tier` | Moves between basic and premium |\n"
            "| `set_account_blocked` | Blocks or unblocks an account |\n"
            "\n"
            "Each one checks that the customer owns what they are acting on, and "
            "refuses an operation that has already happened.\n"
        ),
        md(
            "The server is `agentic/mcp/server.py`. It is **not** flattened into this "
            "notebook, because it runs as its own process — that separation is the "
            "point of putting it behind MCP. For reference:\n"
            "\n"
            "```python\n" + server_source + "```\n"
        ),
        md("The client starts that process over stdio and adapts its tools for LangGraph.\n"),
        code(source_of("agentic/mcp/client.py")),
        code(MCP_ENABLE),
        code(MCP_ATTACH),
    ]


TRY_IT = md("## Try it\n")

cells = [
    md(
        "# UDA-Hub\n"
        "\n"
        "A multi-agent customer support system built with LangGraph.\n"
        "\n"
        "```\n"
        "user -> resolve_context -> load_memory -> classify_urgency\n"
        "                                            |-- escalation ------------.\n"
        "                                            '-- check_knowledge --.    |\n"
        "                                                   |              |    |\n"
        "                                        confident  |              '----+-> escalation\n"
        "                                                   v                   |\n"
        "                                      urgent -> round-robin team       |\n"
        "                                      normal -> single expert          |\n"
        "                                                   |                   |\n"
        "                                                   '-------> finalize <'\n"
        "```\n"
        "\n"
        "A ticket escalates for either of two reasons: the customer asked for a "
        "person, or the knowledge base cannot support an answer.\n"
        "\n"
        "**This notebook is generated from the `agentic/` package by "
        "`build_index_nb.py`.** The package is the code that actually runs; this "
        "notebook is a flattened, readable copy of it. Edit `agentic/` and re-run "
        "the generator rather than editing cells here.\n"
        "\n"
        "The agents here are read-only. `index_mcp.ipynb` is this same notebook with "
        "the MCP operation tools attached, so the agents can also issue refunds, "
        "cancel reservations and change plans.\n"
    ),
    md("## Setup\n"),
    code("# !pip install -r ../requirements.txt\n"),
    code(source_of("agentic/config.py")),
    md(
        "## Logging\n"
        "\n"
        "Every decision the system makes is written as one JSON object per line, so "
        "a whole ticket can be replayed afterwards with `read_log`.\n"
    ),
    code(source_of("agentic/observability.py")),
    md(
        "## Long-term memory\n"
        "\n"
        "Preferences and resolved issues that outlive the session, stored in SQLite. "
        "The other two levels are the ticket record and the checkpointer further "
        "down.\n"
    ),
    code(source_of("agentic/tools/memory.py")),
    md(
        "## Tools\n"
        "\n"
        "### Knowledge base (RAG)\n"
        "\n"
        "`assess_knowledge_confidence` is the gate: it scores whether we have "
        "anything to answer a question with *before* an agent tries. A low score "
        "sends the ticket to a human instead of letting an agent invent an answer.\n"
    ),
    code(source_of("agentic/tools/knowledge.py")),
    md("### Customer data\n"),
    code(source_of("agentic/tools/customer.py")),
    md("### Tickets\n"),
    code(source_of("agentic/tools/tickets.py")),
    md("### The expert toolset\n"),
    code(AGENTIC_TOOLS),
    md("## Classification\n"),
    code(source_of("agentic/tools/classification.py")),
    md("## Agents\n\n### Domain experts\n"),
    code(source_of("agentic/agents/experts.py")),
    md("### Round-robin teams (urgent path)\n"),
    code(source_of("agentic/agents/teams.py")),
    md("### Escalation agent\n"),
    code(source_of("agentic/agents/escalation.py")),
    md("## The orchestrator graph\n"),
    code(source_of("agentic/workflow.py")),
    TRY_IT,
    code(DRAW),
    code(DEMO),
    code(DEMO_URGENT),
    md("### Escalation, both ways in\n"),
    code(DEMO_ESCALATION),
    code(DEMO_UNANSWERABLE),
    md("### The audit trail\n"),
    code(DEMO_LOG),
    md("### Memory across sessions\n"),
    code(DEMO_MEMORY),
]


def write(filename: str, notebook_cells: list) -> None:
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = BASE / filename
    target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target} with {len(notebook_cells)} cells")


write("index.ipynb", cells)

# The MCP notebook is the same system, with the operation tools switched on just
# before the demos, plus one demo that actually changes the account.
split = cells.index(TRY_IT)
write(
    "index_mcp.ipynb",
    cells[:split] + mcp_cells() + cells[split:] + [md("### An operation, end to end\n"), code(MCP_DEMO)],
)
