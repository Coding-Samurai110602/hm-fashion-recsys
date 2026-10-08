"""Single-customer inference path for the H&M recommendation frontend."""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.config import ANCHOR_DATE, ANCHOR_EPOCH_DAYS, MERGE_PRIORITY, PROCESSED_DIR, SOURCE_K
from src.data_io import load_articles, load_customers, load_transactions
from src.features.build import _compute_features_inner
from src.features.registry import FEATURE_LIST
from src.model.data import FEATURE_NAMES
from src.model.state import RecommenderState, build_state


def _week_idx_for_date(d: date) -> int:
    return (d - ANCHOR_DATE).days // 7


class Recommender:
    """Load a trained ranker and serve single-customer recommendations.

    Preloads the full transaction history once. Heavy, customer-independent work
    (article features, popularity lists, copurchase matrix) is computed once per
    as_of_week via build_state() and cached. All per-call work is lightweight
    (small Polars filters + model.predict).

    batch and single-customer paths use identical state and per-customer assembly
    code (via _compute_features_inner with _state), guaranteeing identical results.
    """

    def __init__(
        self,
        model_path: str | Path,
        transactions_path: str | Path | None = None,
        articles_path: str | Path | None = None,
        customers_path: str | Path | None = None,
    ) -> None:
        model_path = Path(model_path)
        self.booster = lgb.Booster(model_file=str(model_path))

        tx_path = Path(transactions_path) if transactions_path else PROCESSED_DIR / "transactions_train.parquet"
        art_path = Path(articles_path) if articles_path else PROCESSED_DIR / "articles.parquet"
        cust_path = Path(customers_path) if customers_path else PROCESSED_DIR / "customers.parquet"

        # Collect full transaction history once; add week_idx
        self._transactions_df: pl.DataFrame = (
            pl.read_parquet(tx_path)
            .with_columns(
                ((pl.col("t_dat").cast(pl.Int32) - ANCHOR_EPOCH_DAYS) // 7)
                .cast(pl.Int16).alias("week_idx")
            )
        )
        self._articles_lf: pl.LazyFrame = pl.scan_parquet(art_path)
        self._customers_lf: pl.LazyFrame = pl.scan_parquet(cust_path)

        # Cache of as_of_week → RecommenderState (built lazily on first recommend() call)
        self._state_cache: dict[int, RecommenderState] = {}

    @property
    def rss_mb(self) -> float:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    def build_state(self, as_of_week: int) -> RecommenderState:
        """Precompute customer-independent state for the given cutoff week.

        Results are cached: calling this twice with the same as_of_week is free.
        Call this explicitly before the first recommend() to amortise latency.
        """
        if as_of_week not in self._state_cache:
            history_df = self._transactions_df.filter(pl.col("week_idx") < as_of_week)
            self._state_cache[as_of_week] = build_state(
                as_of_week, history_df, self._articles_lf, self._customers_lf
            )
        return self._state_cache[as_of_week]

    def _get_state(self, as_of_week: int) -> RecommenderState:
        """Return cached state, building it if needed."""
        return self.build_state(as_of_week)

    def recommend(
        self,
        customer_idx: int,
        as_of_week: int,
        k: int = 12,
    ) -> list[tuple[int, float, list[str]]]:
        """Recommend top-k articles for one real customer.

        Uses precomputed state for all customer-independent work (article features,
        popularity lists, copurchase matrix). Per-call cost is O(customer_history × candidates).

        Parameters
        ----------
        customer_idx: Integer index of the customer.
        as_of_week: Cutoff week index (exclusive).
        k: Number of recommendations.

        Returns
        -------
        List of (article_idx, score, sources) sorted by descending score.
        """
        state = self._get_state(as_of_week)

        # history_lf = state history (already filtered to week_idx < as_of_week)
        history_lf = state.history_df.lazy()

        df_feat = _compute_features_inner(
            history_lf=history_lf,
            ground_truth={customer_idx: set()},
            eval_customers=[customer_idx],
            articles_df=state.articles_lf,
            customers_df=state.customers_df.lazy(),
            fold_week=as_of_week,
            neg_sample_rate=1.0,
            seed=42,
            _state=state,
        )

        if df_feat.is_empty():
            return []

        df_cust = df_feat.filter(pl.col("customer_idx") == customer_idx)
        if df_cust.is_empty():
            return []

        X = (
            df_cust.select(FEATURE_NAMES)
            .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
            .to_numpy(allow_copy=True)
        )
        scores = self.booster.predict(X)

        source_cols = [f"in_{s}" for s in MERGE_PRIORITY]
        article_list = df_cust["article_idx"].to_list()
        sources_arr = df_cust.select(source_cols).to_numpy(allow_copy=True)

        order = np.argsort(-scores)
        results = []
        for i in order[:k]:
            idx = int(i)
            src_list = [MERGE_PRIORITY[j] for j, v in enumerate(sources_arr[idx]) if v == 1]
            results.append((article_list[idx], float(scores[idx]), src_list))

        return results

    def recommend_custom(
        self,
        history_rows: list[dict],
        as_of_week: int,
        k: int = 12,
    ) -> list[tuple[int, float, list[str]]]:
        """Recommend for a synthetic customer with hand-built purchase history.

        Uses precomputed state for article features, popularity lists, and the
        copurchase symmetric matrix. Customer-level and interaction features are
        computed from the supplied history.

        Parameters
        ----------
        history_rows: List of dicts with keys:
            - article_idx (int)
            - t_dat (date or str YYYY-MM-DD)
            - price (float, optional; default 0.05)
            - sales_channel_id (int, optional; default 2)
        as_of_week: Cutoff week index.
        k: Number of recommendations.
        """
        state = self._get_state(as_of_week)

        synthetic_idx = -1  # does not collide with real int32 indices

        rows = []
        for r in history_rows:
            t_dat = r["t_dat"]
            if isinstance(t_dat, str):
                t_dat = date.fromisoformat(t_dat)
            week_idx = (t_dat - ANCHOR_DATE).days // 7
            if week_idx >= as_of_week:
                continue  # leakage guard: skip future purchases
            rows.append({
                "customer_idx": synthetic_idx,
                "article_idx": int(r["article_idx"]),
                "t_dat": t_dat,
                "price": float(r.get("price", 0.05)),
                "sales_channel_id": int(r.get("sales_channel_id", 2)),
                "week_idx": week_idx,
            })

        if not rows:
            return []

        history_df = pl.DataFrame(
            rows,
            schema={
                "customer_idx": pl.Int32,
                "article_idx": pl.Int32,
                "t_dat": pl.Date,
                "price": pl.Float32,
                "sales_channel_id": pl.Int8,
                "week_idx": pl.Int16,
            },
        )

        # Combine real history + synthetic customer rows
        real_cols = ["customer_idx", "article_idx", "t_dat", "price", "sales_channel_id", "week_idx"]
        combined_df = pl.concat(
            [state.history_df.select(real_cols), history_df.select(real_cols)],
            how="vertical",
        )
        combined_lf = combined_df.lazy()

        df_feat = _compute_features_inner(
            history_lf=combined_lf,
            ground_truth={synthetic_idx: set()},
            eval_customers=[synthetic_idx],
            articles_df=state.articles_lf,
            customers_df=state.customers_df.lazy(),
            fold_week=as_of_week,
            neg_sample_rate=1.0,
            seed=42,
            _state=state,
        )

        if df_feat.is_empty():
            return []

        df_cust = df_feat.filter(pl.col("customer_idx") == synthetic_idx)
        if df_cust.is_empty():
            return []

        X = (
            df_cust.select(FEATURE_NAMES)
            .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
            .to_numpy(allow_copy=True)
        )
        scores = self.booster.predict(X)

        source_cols = [f"in_{s}" for s in MERGE_PRIORITY]
        article_list = df_cust["article_idx"].to_list()
        sources_arr = df_cust.select(source_cols).to_numpy(allow_copy=True)

        order = np.argsort(-scores)
        results = []
        for i in order[:k]:
            idx = int(i)
            src_list = [MERGE_PRIORITY[j] for j, v in enumerate(sources_arr[idx]) if v == 1]
            results.append((article_list[idx], float(scores[idx]), src_list))

        return results

    def latency_benchmark(
        self,
        customer_indices: list[int],
        as_of_week: int,
        warmup: int = 5,
    ) -> dict:
        """Measure p50/p95 single-customer call latency after warm-up.

        Parameters
        ----------
        customer_indices: Customers to benchmark (after warm-up).
        as_of_week: Cutoff week.
        warmup: Number of warm-up calls (state is pre-built before timing starts).

        Returns
        -------
        dict with p50_ms, p95_ms, n_calls, rss_mb, state_build_time_s, state_rss_mb.
        """
        state = self.build_state(as_of_week)

        for c in customer_indices[:warmup]:
            self.recommend(c, as_of_week)

        latencies_ms = []
        for c in customer_indices:
            t0 = time.perf_counter()
            self.recommend(c, as_of_week)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        return {
            "p50_ms": float(np.percentile(latencies_ms, 50)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "n_calls": len(latencies_ms),
            "rss_mb": self.rss_mb,
            "state_build_time_s": state.build_time_s,
            "state_rss_mb": state.rss_mb,
        }
