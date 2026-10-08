"""Schema tests for feature tables.

Verifies that fold parquet outputs have:
- Exactly the columns from FEATURE_LIST in order
- Correct dtypes per FEATURE_LIST
- No float64 or string columns
- (customer_idx, article_idx) unique per fold
"""
import pytest
import polars as pl
from pathlib import Path

from src.config import HOLDOUT_WEEK, PROCESSED_DIR, VALIDATION_WEEKS
from src.features.registry import FEATURE_LIST

FEATURES_DIR = PROCESSED_DIR / "features"


def _available_folds() -> list[int]:
    """Return fold weeks that have been built and exist on disk."""
    folds = []
    for w in list(VALIDATION_WEEKS) + [HOLDOUT_WEEK]:
        if (FEATURES_DIR / f"fold_{w}.parquet").exists():
            folds.append(w)
    return folds


def test_holdout_labels_written_separately():
    """Verify labels_104.parquet exists and has the right schema."""
    labels_path = PROCESSED_DIR / "features" / "labels_104.parquet"
    if not labels_path.exists():
        pytest.skip("labels_104.parquet not yet built")
    labels = pl.read_parquet(labels_path)
    assert "customer_idx" in labels.columns
    assert "article_idx" in labels.columns
    assert "label" in labels.columns
    assert labels["label"].dtype == pl.Int8
    assert set(labels["label"].unique().to_list()).issubset({0, 1})


@pytest.mark.parametrize("fold_week", _available_folds())
class TestFoldSchema:

    def _load(self, fold_week: int) -> pl.DataFrame:
        return pl.read_parquet(FEATURES_DIR / f"fold_{fold_week}.parquet")

    def test_columns_match_feature_list_in_order(self, fold_week):
        df = self._load(fold_week)
        expected_feature_cols = [spec.name for spec in FEATURE_LIST]
        expected_all = ["customer_idx", "article_idx", "label"] + expected_feature_cols
        assert list(df.columns) == expected_all, (
            f"Fold {fold_week}: column mismatch.\n"
            f"  Expected: {expected_all}\n"
            f"  Got:      {list(df.columns)}"
        )

    def test_feature_dtypes_match_registry(self, fold_week):
        df = self._load(fold_week)
        mismatches = []
        for spec in FEATURE_LIST:
            if spec.name not in df.columns:
                mismatches.append(f"{spec.name}: MISSING")
                continue
            actual = df[spec.name].dtype
            expected = spec.dtype
            if actual != expected:
                mismatches.append(f"{spec.name}: expected {expected}, got {actual}")
        assert not mismatches, f"Fold {fold_week} dtype mismatches:\n" + "\n".join(mismatches)

    def test_no_float64_in_any_column(self, fold_week):
        df = self._load(fold_week)
        float64_cols = [c for c in df.columns if df[c].dtype == pl.Float64]
        assert not float64_cols, f"Fold {fold_week}: float64 columns found: {float64_cols}"

    def test_no_string_columns_in_features(self, fold_week):
        df = self._load(fold_week)
        str_cols = [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.Categorical)]
        assert not str_cols, f"Fold {fold_week}: string/categorical columns found: {str_cols}"

    def test_key_pair_unique(self, fold_week):
        df = self._load(fold_week)
        n_rows = len(df)
        n_unique_pairs = df.select(["customer_idx", "article_idx"]).n_unique()
        assert n_rows == n_unique_pairs, (
            f"Fold {fold_week}: (customer_idx, article_idx) not unique. "
            f"rows={n_rows}, unique_pairs={n_unique_pairs}"
        )

    def test_key_dtypes(self, fold_week):
        df = self._load(fold_week)
        assert df["customer_idx"].dtype == pl.Int32, \
            f"Fold {fold_week}: customer_idx dtype should be Int32"
        assert df["article_idx"].dtype == pl.Int32, \
            f"Fold {fold_week}: article_idx dtype should be Int32"
        assert df["label"].dtype == pl.Int8, \
            f"Fold {fold_week}: label dtype should be Int8"

    def test_label_values_valid(self, fold_week):
        df = self._load(fold_week)
        if fold_week == HOLDOUT_WEEK:
            # Holdout: label is -1 placeholder
            unique_labels = df["label"].unique().to_list()
            assert unique_labels == [-1], f"Fold 104 labels should all be -1, got {unique_labels}"
        else:
            unique_labels = set(df["label"].drop_nulls().to_list())
            assert unique_labels.issubset({0, 1}), \
                f"Fold {fold_week}: label values should be 0 or 1, got {unique_labels}"

    def test_positives_exist_in_validation_folds(self, fold_week):
        if fold_week == HOLDOUT_WEEK:
            return
        df = self._load(fold_week)
        n_pos = int((df["label"] == 1).sum())
        assert n_pos > 0, f"Fold {fold_week}: no positives found"

