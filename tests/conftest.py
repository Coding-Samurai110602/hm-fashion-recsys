"""Top-level pytest configuration: auto-skip data-dependent tests in CI.

Tests marked with @pytest.mark.requires_data are automatically skipped when
the processed Kaggle data is absent.  The data root is controlled by the
HM_DATA_DIR environment variable (default: data/).
"""
import os
from pathlib import Path

import pytest


def _processed_data_missing() -> bool:
    data_root = Path(os.environ.get("HM_DATA_DIR", "data"))
    return not (data_root / "processed" / "transactions_train.parquet").exists()


def pytest_collection_modifyitems(config, items):
    if not _processed_data_missing():
        return
    skip = pytest.mark.skip(
        reason=(
            "Kaggle data not available — set HM_DATA_DIR or run "
            "scripts/convert_to_parquet.py first"
        )
    )
    for item in items:
        if item.get_closest_marker("requires_data") and not item.get_closest_marker(
            "not_requires_data"
        ):
            item.add_marker(skip)
