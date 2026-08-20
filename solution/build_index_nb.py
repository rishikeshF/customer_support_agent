"""One-off generator: rebuilds index.ipynb from the agentic/ module sources.

Keeping the notebook generated rather than hand-copied means it cannot drift
away from the code that actually runs. Re-run it after changing agentic/.
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
    md("## Try it\n"),
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

notebook = {
    "cells": cells,
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

target = BASE / "index.ipynb"
target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"wrote {target} with {len(cells)} cells")
