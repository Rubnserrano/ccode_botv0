"""Shared fixtures for store tests."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_store_data():
    """Remove test_exchange/TESTSYM data before each store test."""
    base = Path("data/raw") / "test_exchange" / "testsym"
    if base.exists():
        shutil.rmtree(base)
