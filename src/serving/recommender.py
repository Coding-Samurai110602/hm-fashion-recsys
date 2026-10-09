"""Bundle-based recommender: all inference from precomputed state in the bundle.

No raw transaction scans. Everything is served from the bundle's precomputed parquet files.

API:
  recommend(customer_idx)         → 12 items with reasons
  recommend_custom(history_rows)  → same, for synthetic history
  explain(customer_idx, article_idx) → full contrast-SHAP breakdown
  baseline(customer_idx)          → heuristic top-12 (for side-by-side)
  candidate_funnel(customer_idx)  → all candidates with source/score/rank
  actual_purchases(customer_idx)  → week-104 purchases (display only)
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl

from src.model.data import FEATURE_NAMES
from src.config import MERGE_PRIORITY


def _age_bucket(age: float | None) -> str:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "missing"
    if age < 25:
        return "<25"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


class BundleRecommender:
    """Serve recommendations from a loaded Bundle, with no raw-data scans.

    All candidate generation and feature assembly reuse the state parquets
    stored in the bundle. Contrast-SHAP explanations are computed on-the-fly.
    """

    def __init__(self, bundle: "Bundle") -> None:  # noqa: F821
        from src.serving.bundle import Bundle
        assert isinstance(bundle, Bundle)
        self._b = bundle
        self._booster = bundle.booster
        self._as_of_week = bundle.manifest.get("as_of_week", 104)

        # Build the state ONCE from bundle data; reuse for every call
        self._state = self._make_state_from_bundle()

        # Customer history index is already in state.history_df; no extra dict needed

    # ── Public API ─────────────────────────────────────────────────────────

    def recommend(
        self,
        customer_idx: int,
        k: int = 12,
    ) -> list[dict[str, Any]]:
        """Recommend top-k items for a real customer.

        Returns list of dicts:
          article_idx, article_id, prod_name, product_type, colour, department,
          score, sources, top_3_reasons
        """
        df_feat, scores, sources_arr = self._score_customer(customer_idx)
        if df_feat is None:
            return []
        return self._format_results(df_feat, scores, sources_arr, k, customer_idx)

    def recommend_custom(
        self,
        history_rows: list[dict],
        k: int = 12,
    ) -> list[dict[str, Any]]:
        """Recommend for a synthetic customer with hand-built history.

        history_rows: list of dicts with keys: article_idx, t_dat (YYYY-MM-DD), price (opt), sales_channel_id (opt)
        """
        from datetime import date as date_type
        from src.config import ANCHOR_DATE

        synthetic_idx = -1
        rows = []
        for r in history_rows:
            t_dat = r["t_dat"]
            if isinstance(t_dat, str):
                t_dat = date_type.fromisoformat(t_dat)
            week_idx = (t_dat - ANCHOR_DATE).days // 7
            if week_idx >= self._as_of_week:
                continue
            rows.append({
                "customer_idx": synthetic_idx,
                "article_idx": int(r["article_idx"]),
                "t_dat": t_dat,
                "price": float(r.get("price", 0.05)),
                "sales_channel_id": int(r.get("sales_channel_id", 2)),
                "week_idx": week_idx,
            })
        if not rows:
            # No valid history — return top-k from popularity
            return self._popularity_fallback(k)

        hist_df = pl.DataFrame(
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
        df_feat, scores, sources_arr = self._score_customer(
            synthetic_idx, custom_history=hist_df
        )

        if df_feat is None:
            return []
        return self._format_results(df_feat, scores, sources_arr, k, synthetic_idx)

    def explain(
        self,
        customer_idx: int,
        article_idx: int,
    ) -> dict[str, Any]:
        """Full contrast-SHAP breakdown for one (customer, article) pair.

        Returns dict with:
          customer_idx, article_idx, score, expected_value,
          shap_values: {feature_name: shap_val},
          contrast: {feature_name: contrast_val},
          reasons: [str]
        """
        from src.explain.shap_values import compute_shap, compute_within_list_contrast
        from src.explain.reasons import generate_reasons

        df_feat, scores, _ = self._score_customer(customer_idx)
        if df_feat is None:
            return {"error": "customer not found"}

        # Find the row for this article
        art_mask = (df_feat["article_idx"] == article_idx).to_numpy()
        if not art_mask.any():
            return {"error": "article not in candidate pool"}

        X = (
            df_feat.select(FEATURE_NAMES)
            .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
            .to_numpy(allow_copy=True)
        ).astype(np.float32)

        shap_vals, expected_val, _ = compute_shap(self._booster, X)
        custs = df_feat["customer_idx"].to_numpy()
        contrast = compute_within_list_contrast(shap_vals, custs)

        row_i = int(np.where(art_mask)[0][0])
        shap_row = shap_vals[row_i]
        contrast_row = contrast[row_i]
        fv_row = X[row_i]

        reasons = generate_reasons(contrast_row, fv_row, FEATURE_NAMES, top_k=3)

        return {
            "customer_idx": customer_idx,
            "article_idx": article_idx,
            "score": float(scores[row_i]),
            "expected_value": float(expected_val[row_i]),
            "shap_values": {
                n: round(float(v), 8) for n, v in zip(FEATURE_NAMES, shap_row)
            },
            "contrast": {
                n: round(float(v), 8) for n, v in zip(FEATURE_NAMES, contrast_row)
            },
            "reasons": reasons,
        }

    def baseline(
        self,
        customer_idx: int,
        k: int = 12,
    ) -> list[dict[str, Any]]:
        """Heuristic top-12: recency-ordered repurchase + popularity fill."""
        b = self._b
        state = self._state
        hist = state.history_df.filter(
            pl.col("customer_idx") == int(customer_idx)
        ) if state.history_df is not None else None
        results = []

        if hist is not None and len(hist) > 0:
            # Repurchase: most recent first, then article_idx ASC (Baseline B ordering)
            recent = (
                hist
                .group_by("article_idx")
                .agg(pl.col("t_dat").max().alias("last_date"))
                .sort(["last_date", "article_idx"], descending=[True, False])
                .head(k)
            )
            for row in recent.iter_rows(named=True):
                art_idx = int(row["article_idx"])
                meta = self._get_article_meta(art_idx)
                results.append({**meta, "score": None, "sources": ["repurchase"], "top_3_reasons": []})

        # Fill remaining slots from popularity
        filled = {r["article_idx"] for r in results}
        if len(results) < k and b.pop_last_week is not None:
            for row in b.pop_last_week.iter_rows(named=True):
                if len(results) >= k:
                    break
                art_idx = int(row["article_idx"])
                if art_idx not in filled:
                    filled.add(art_idx)
                    meta = self._get_article_meta(art_idx)
                    results.append({
                        **meta,
                        "score": float(row.get("score", 0.0)),
                        "sources": ["popularity_last_week"],
                        "top_3_reasons": [],
                    })

        return results[:k]

    def candidate_funnel(
        self,
        customer_idx: int,
    ) -> list[dict[str, Any]]:
        """All candidates with source, source rank, model score, and final rank."""
        df_feat, scores, sources_arr = self._score_customer(customer_idx)
        if df_feat is None:
            return []

        order = np.argsort(-scores)
        results = []
        for final_rank, i in enumerate(order, start=1):
            art_idx = int(df_feat["article_idx"][int(i)])
            src_list = [MERGE_PRIORITY[j] for j, v in enumerate(sources_arr[int(i)]) if v == 1]
            final_rank_feat = float(df_feat["final_rank"][int(i)]) if "final_rank" in df_feat.columns else None
            results.append({
                "article_idx": art_idx,
                "source": src_list[0] if src_list else "unknown",
                "all_sources": src_list,
                "source_rank": final_rank_feat,
                "model_score": float(scores[int(i)]),
                "final_rank": final_rank,
            })
        return results

    def actual_purchases(
        self,
        customer_idx: int,
    ) -> list[dict[str, Any]]:
        """Week-104 actual purchases for display only (clearly labelled, not for evaluation)."""
        b = self._b
        if b.gt_week104 is None:
            return []
        gt_cust = b.gt_week104.filter(pl.col("customer_idx") == customer_idx)
        results = []
        for row in gt_cust.iter_rows(named=True):
            art_idx = int(row["article_idx"])
            meta = self._get_article_meta(art_idx)
            results.append({**meta, "_display_only_ground_truth": True})
        return results

    # ── Internal helpers ────────────────────────────────────────────────────

    def _score_customer(
        self,
        customer_idx: int,
        custom_history: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame | None, np.ndarray | None, np.ndarray | None]:
        """Build feature table and score for one customer from bundle state.

        Uses the cached state (built once in __init__). No raw transaction scans.
        """
        from src.features.build import _compute_features_inner

        state = self._state

        if custom_history is not None:
            # Merge synthetic history with the bundle's history
            real_cols = ["customer_idx", "article_idx", "t_dat", "price",
                         "sales_channel_id", "week_idx"]
            combined = pl.concat(
                [state.history_df.select(real_cols), custom_history.select(real_cols)],
                how="vertical",
            )
            history_lf = combined.lazy()
        else:
            history_lf = state.history_df.lazy()

        df_feat = _compute_features_inner(
            history_lf=history_lf,
            ground_truth={customer_idx: set()},
            eval_customers=[int(customer_idx)],
            articles_df=state.articles_lf,
            customers_df=state.customers_df.lazy(),
            fold_week=self._as_of_week,
            neg_sample_rate=1.0,
            seed=42,
            _state=state,
        )

        if df_feat is None or df_feat.is_empty():
            return None, None, None

        df_cust = df_feat.filter(pl.col("customer_idx") == customer_idx)
        if df_cust.is_empty():
            return None, None, None

        X = (
            df_cust.select(FEATURE_NAMES)
            .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
            .to_numpy(allow_copy=True)
        ).astype(np.float32)

        scores = self._booster.predict(X).astype(np.float32)
        source_cols = [f"in_{s}" for s in MERGE_PRIORITY]
        sources_arr = df_cust.select(source_cols).to_numpy(allow_copy=True)

        return df_cust, scores, sources_arr

    def _popularity_fallback(self, k: int = 12) -> list[dict]:
        """Return top-k from last-week popularity (used when history is unavailable)."""
        b = self._b
        if b.pop_last_week is None:
            return []
        results = []
        for row in b.pop_last_week.head(k).iter_rows(named=True):
            art_idx = int(row["article_idx"])
            meta = self._get_article_meta(art_idx)
            results.append({
                **meta,
                "score": float(row.get("score", 0.0)),
                "sources": ["popularity_last_week"],
                "top_3_reasons": [],
            })
        return results

    def _make_state_from_bundle(self):
        """Wrap bundle data in a RecommenderState for _compute_features_inner.

        All data comes from the bundle's precomputed parquets — no raw scans.
        """
        from src.model.state import RecommenderState
        b = self._b
        # Use full customers table if available; fall back to age_buckets only
        customers_df = b.customers if b.customers is not None else (
            b.age_buckets if b.age_buckets is not None else pl.DataFrame()
        )
        # articles_lf must be the raw articles table (for compute_interaction_features:
        # product_code, product_group_name, garment_group_name)
        articles_lf = (
            b.articles_raw.lazy() if b.articles_raw is not None
            else b.article_feats.lazy()  # fallback: may not have all columns
        )
        return RecommenderState(
            as_of_week=self._as_of_week,
            history_df=b.customer_history,
            article_feats=b.article_feats,
            customers_df=customers_df,
            articles_lf=articles_lf,
            all_cust_buckets=b.age_buckets,
            bucket_totals=b.bucket_totals,
            pop_last_week_top=b.pop_last_week,
            pop_decayed_top=b.pop_decayed,
            seg_pop_by_bucket=b.seg_pop_by_bucket,
            copurchase_symmetric=b.copurchase,
            prev_week_sales=b.prev_week_sales,
            article_product_codes=b.article_product_codes,
            total_tx_last_week=b.total_tx_last_week,
            total_decayed_tx=b.total_decayed_tx,
            build_time_s=0.0,
            rss_mb=0.0,
        )

    def _format_results(
        self,
        df_cust: pl.DataFrame,
        scores: np.ndarray,
        sources_arr: np.ndarray,
        k: int,
        customer_idx: int,
    ) -> list[dict[str, Any]]:
        """Format scored candidates into the recommendation output schema."""
        from src.explain.shap_values import compute_shap, compute_within_list_contrast
        from src.explain.reasons import generate_reasons

        X = (
            df_cust.select(FEATURE_NAMES)
            .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
            .to_numpy(allow_copy=True)
        ).astype(np.float32)

        shap_vals, _, _ = compute_shap(self._booster, X)
        custs = df_cust["customer_idx"].to_numpy()
        contrast = compute_within_list_contrast(shap_vals, custs)

        order = np.argsort(-scores)
        results = []
        for i in order[:k]:
            idx = int(i)
            art_idx = int(df_cust["article_idx"][idx])
            src_list = [MERGE_PRIORITY[j] for j, v in enumerate(sources_arr[idx]) if v == 1]

            reasons = generate_reasons(contrast[idx], X[idx], FEATURE_NAMES, top_k=3)
            meta = self._get_article_meta(art_idx)

            results.append({
                **meta,
                "score": float(scores[idx]),
                "sources": src_list,
                "top_3_reasons": reasons,
            })

        return results

    def _get_article_meta(self, article_idx: int) -> dict[str, Any]:
        """Look up display metadata for one article."""
        b = self._b
        if b.article_meta is None:
            return {"article_idx": article_idx, "article_id": str(article_idx),
                    "prod_name": "", "product_type": "", "colour": "", "department": ""}
        row = b.article_meta.filter(pl.col("article_idx") == article_idx)
        if len(row) == 0:
            return {"article_idx": article_idx, "article_id": str(article_idx),
                    "prod_name": "", "product_type": "", "colour": "", "department": ""}
        return {
            "article_idx": article_idx,
            "article_id": str(row["article_id"][0]),
            "prod_name": str(row.get_column("prod_name")[0]) if "prod_name" in row.columns else "",
            "product_type": str(row.get_column("product_type_name")[0]) if "product_type_name" in row.columns else "",
            "colour": str(row.get_column("colour_group_name")[0]) if "colour_group_name" in row.columns else "",
            "department": str(row.get_column("department_name")[0]) if "department_name" in row.columns else "",
        }
