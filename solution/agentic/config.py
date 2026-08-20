"""Shared configuration: paths, model clients and the vector store.

Everything that needs an API key or a file path is created once here, so the
tools and agents can just import what they need.
"""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# Paths are resolved from this file, so the code runs from any folder.
# `__file__` is absent when this module is flattened into a notebook cell, in
# which case the notebook already runs from solution/.
BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
CORE_DB = BASE_DIR / "data" / "core" / "udahub.db"
EXTERNAL_DB = BASE_DIR / "data" / "external" / "cultpass.db"
VECTOR_DIR = BASE_DIR / "data" / "vectorstore"
LOG_DIR = BASE_DIR / "data" / "logs"
CHECKPOINT_DB = BASE_DIR / "data" / "core" / "checkpoints.db"

# How sure we need to be that the knowledge base covers a question before we
# let an agent answer it. Below this, the ticket goes to a human instead.
KNOWLEDGE_CONFIDENCE_THRESHOLD = 0.5

# Prefer a real OpenAI key when there is one, and fall back to the Vocareum
# proxy. Setting OPENAI_BASE_URL overrides either choice.
if os.getenv("OPENAI_API_KEY"):
    API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
else:
    API_KEY = os.getenv("VOCAREUM_API_KEY")
    DEFAULT_BASE_URL = "https://openai.vocareum.com/v1"

BASE_URL = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url=BASE_URL,
    api_key=API_KEY,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=BASE_URL,
    api_key=API_KEY,
)

vectorstore = FAISS.load_local(
    str(VECTOR_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection whose rows behave like dictionaries."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def build_checkpointer() -> SqliteSaver:
    """
    Short-term memory, backed by a file so it survives a restart.

    `SqliteSaver.from_conn_string` is a context manager and closes the
    connection on exit, so we own the connection ourselves instead. The graph
    is only ever used from the notebook or a script, hence check_same_thread.
    """
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
