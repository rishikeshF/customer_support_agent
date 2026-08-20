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

load_dotenv()

# Paths are resolved from this file, so the code runs from any folder.
# `__file__` is absent when this module is flattened into a notebook cell, in
# which case the notebook already runs from solution/.
BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
CORE_DB = BASE_DIR / "data" / "core" / "udahub.db"
EXTERNAL_DB = BASE_DIR / "data" / "external" / "cultpass.db"
VECTOR_DIR = BASE_DIR / "data" / "vectorstore"

API_KEY = os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

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
