from __future__ import annotations

from pathlib import Path

import pytest


TEST_RESULTS_DIR = Path(__file__).resolve().parents[1] / "TestResults"


@pytest.fixture(scope="session")
def test_results_dir() -> Path:
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_RESULTS_DIR