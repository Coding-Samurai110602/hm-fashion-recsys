"""LightGBM LambdaRank training: inner-loop (folds 100-101 → val 102) and final model."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from src.config import HOLDOUT_WEEK, VALIDATION_WEEKS
from src.model.data import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    concat_folds,
    prepare_dataset,
)

# Fixed seed and threads for reproducibility
SEED: int = 42
NUM_THREADS: int = 4

# Default parameters (objective + stable values)
DEFAULT_PARAMS: dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [12],
    "lambdarank_truncation_level": 12,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_jobs": NUM_THREADS,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "seed": SEED,
    "data_random_seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "drop_seed": SEED,
}


def _build_lgb_dataset(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    reference: lgb.Dataset | None = None,
    free_raw_data: bool = True,
) -> lgb.Dataset:
    """Wrap arrays in lgb.Dataset with categorical feature declarations."""
    cat_indices = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]
    ds = lgb.Dataset(
        X,
        label=y,
        group=groups,
        categorical_feature=cat_indices,
        free_raw_data=free_raw_data,
        reference=reference,
    )
    return ds


def train_inner(
    params: dict | None = None,
    n_rounds: int = 500,
    early_stopping_rounds: int = 30,
    verbose_eval: int = 50,
) -> tuple[lgb.Booster, int, float]:
    """Train on folds 100+101, early-stop on fold 102.

    Returns the trained booster, the stopping round, and the best NDCG@12
    on fold 102.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    df_train = concat_folds([100, 101])
    df_val = concat_folds([102])

    X_train, y_train, g_train = prepare_dataset(df_train, drop_zero_positive_groups=True)
    X_val, y_val, g_val = prepare_dataset(df_val, drop_zero_positive_groups=False)

    ds_train = _build_lgb_dataset(X_train, y_train, g_train, free_raw_data=True)
    ds_val = _build_lgb_dataset(X_val, y_val, g_val, reference=ds_train, free_raw_data=True)

    callbacks = [
        lgb.log_evaluation(period=verbose_eval),
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
    ]

    booster = lgb.train(
        params,
        ds_train,
        num_boost_round=n_rounds,
        valid_sets=[ds_val],
        valid_names=["val_102"],
        callbacks=callbacks,
    )

    best_round = booster.best_iteration
    best_score = booster.best_score["val_102"].get("ndcg@12", float("nan"))
    return booster, best_round, best_score


def train_final(
    params: dict,
    inner_rounds: int,
    fold_train: list[int] | None = None,
) -> tuple[lgb.Booster, int, float, str]:
    """Retrain on folds 100-102 with scaled boosting rounds.

    Scaling rule: final_rounds = round(inner_rounds * (n_rows_final / n_rows_inner)),
    where n_rows = actual training rows after zero-positive group removal.
    Approximated as inner_rounds * (rows_100+101+102 / rows_100+101). Minimum 10.
    """
    if fold_train is None:
        fold_train = [100, 101, 102]

    # Load the inner-loop training folds to estimate the size ratio
    df_inner = concat_folds([100, 101])
    df_final = concat_folds(fold_train)

    # Drop zero-positive groups to get effective row counts
    _, _, g_inner = prepare_dataset(df_inner, drop_zero_positive_groups=True)
    _, _, g_final = prepare_dataset(df_final, drop_zero_positive_groups=True)

    n_inner = int(g_inner.sum())
    n_final = int(g_final.sum())
    ratio = n_final / n_inner if n_inner > 0 else 1.5
    final_rounds = max(10, round(inner_rounds * ratio))

    X_final, y_final, g_final_arr = prepare_dataset(df_final, drop_zero_positive_groups=True)
    ds_final = _build_lgb_dataset(X_final, y_final, g_final_arr, free_raw_data=True)

    booster = lgb.train(
        params,
        ds_final,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(period=50)],
    )

    scale_rule = "final_rounds = round(inner_rounds * n_rows_final / n_rows_inner)"
    return booster, final_rounds, round(ratio, 4), scale_rule


def get_feature_importance(booster: lgb.Booster, importance_type: str = "gain") -> list[dict]:
    """Return feature importances sorted descending."""
    vals = booster.feature_importance(importance_type=importance_type)
    return sorted(
        [{"feature": n, "importance": float(v)} for n, v in zip(FEATURE_NAMES, vals)],
        key=lambda x: -x["importance"],
    )
