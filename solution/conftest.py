"""Pytest configuration: makes the model-calling tests opt-in via --llm."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--llm",
        action="store_true",
        default=False,
        help="also run the tests that call the language model (slow, uses quota)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: test calls the language model")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--llm"):
        return
    skip = pytest.mark.skip(reason="needs --llm")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
