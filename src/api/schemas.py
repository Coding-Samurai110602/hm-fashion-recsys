"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator

# ── Shared ────────────────────────────────────────────────────────────────────


class ArticleMeta(BaseModel):
    article_idx: int
    article_id: str
    prod_name: str
    product_type: str
    colour: str
    department: str


class RecommendedItem(BaseModel):
    rank: int
    article_idx: int
    article_id: str
    prod_name: str
    product_type: str
    colour: str
    department: str
    sources: list[str]
    reasons: list[str]


class RecommendedItemNoReasons(BaseModel):
    rank: int
    article_idx: int
    article_id: str
    prod_name: str
    product_type: str
    colour: str
    department: str
    sources: list[str]


class FunnelItem(BaseModel):
    article_idx: int
    source: str
    all_sources: list[str]
    source_rank: float | None
    model_score: float
    final_rank: int


class ExplainResult(BaseModel):
    customer_idx: int
    article_idx: int
    score: float
    expected_value: float
    shap_values: dict[str, float]
    contrast: dict[str, float]
    reasons: list[str]


# ── Customer endpoints ─────────────────────────────────────────────────────────


class CustomerProfile(BaseModel):
    customer_idx: int
    customer_id: str
    age_bucket: str
    n_purchases_all_time: int
    n_purchases_last_4w: int
    recent_history: list[ArticleMeta]


class SampleCustomer(BaseModel):
    customer_idx: int
    customer_id: str
    segment: str
    age_bucket: str


# ── Custom recommendations ─────────────────────────────────────────────────────


class CustomHistoryItem(BaseModel):
    article_id: str
    t_dat: date
    price: float | None = None
    sales_channel_id: int | None = None

    @field_validator("t_dat")
    @classmethod
    def no_future_date(cls, v: date) -> date:
        from datetime import date as date_type

        # Bundle cutoff is week 104 (2020-09-22); reject dates after that
        cutoff = date_type(2020, 9, 22)
        if v > cutoff:
            raise ValueError(f"t_dat {v} is after bundle cutoff {cutoff}")
        return v


class CustomRecommendRequest(BaseModel):
    history: list[CustomHistoryItem] = Field(..., min_length=1)


# ── Experiment endpoints ───────────────────────────────────────────────────────


class PowerRequest(BaseModel):
    baseline_rate: float = Field(..., gt=0, lt=1)
    relative_lift: float = Field(..., gt=0)
    alpha: float = Field(0.05, gt=0, lt=1)
    power: float = Field(0.80, gt=0, lt=1)
    weekly_traffic: int | None = Field(None, gt=0)
    cuped_variance_reduction: float = Field(0.0, ge=0, lt=1)


class PowerResult(BaseModel):
    n_per_arm: int
    n_per_arm_statsmodels: int
    n_per_arm_cuped: int | None
    weeks_needed: float | None
    cohen_h: float


class SimulateRequest(BaseModel):
    test_type: str = Field("ab", pattern="^(ab|aa)$")
    n_per_arm: int = Field(..., gt=0, le=100_000)
    n_sims: int = Field(100, gt=0, le=2000)
    peeking: bool = False
    n_days: int = Field(14, gt=0, le=60)
    seed: int = 42


class SimulateResult(BaseModel):
    test_type: str
    n_per_arm: int
    n_sims: int
    false_positive_rate: float | None
    empirical_power: float | None
    peeking_fpr: float | None


# ── Ops ───────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool


class VersionResponse(BaseModel):
    api_version: str
    git_commit: str
    bundle_as_of_week: int
    model_map12_fold103: float
    model_map12_holdout: float
    n_features: int
