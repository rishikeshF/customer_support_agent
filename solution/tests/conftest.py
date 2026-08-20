"""Pytest configuration: the opt-in test tiers, and log isolation."""

import sys
from pathlib import Path

import pytest

# The tests import `index` and `agentic`, which live one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption(
        "--llm",
        action="store_true",
        default=False,
        help="also run the tests that call the language model (slow, uses quota)",
    )
    parser.addoption(
        "--mcp",
        action="store_true",
        default=False,
        help="also run the tests that start the MCP server as a child process",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: test calls the language model")
    config.addinivalue_line("markers", "mcp: test starts the MCP server as a child process")


def pytest_collection_modifyitems(config, items):
    for flag in ("llm", "mcp"):
        if config.getoption(f"--{flag}"):
            continue
        skip = pytest.mark.skip(reason=f"needs --{flag}")
        for item in items:
            if flag in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """
    Send `log_event` to a throwaway file for the duration of each test.

    The workflow and the logging tests both append to the log, and without this
    every test run would leave its events in the real `data/logs/uda-hub.jsonl`
    alongside the events from actual conversations. Patching the module globals
    (rather than passing a path around) keeps the production code unaware that
    it is under test.
    """
    import agentic.observability as observability

    log_dir = tmp_path / "logs"
    monkeypatch.setattr(observability, "LOG_DIR", log_dir)
    monkeypatch.setattr(observability, "LOG_FILE", log_dir / "uda-hub.jsonl")
    yield log_dir / "uda-hub.jsonl"
