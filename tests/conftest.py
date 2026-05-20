"""Shared fixtures for ts_store tests."""
import shutil
from pathlib import Path

import pytest

TEST_ASSET_ID = "market:test_exchange:testsym"
TEST_BASE = Path("data/ts/market") / TEST_ASSET_ID


@pytest.fixture(autouse=True)
def _clean_test_store():
    if TEST_BASE.exists():
        shutil.rmtree(TEST_BASE)
    yield
    if TEST_BASE.exists():
        shutil.rmtree(TEST_BASE)