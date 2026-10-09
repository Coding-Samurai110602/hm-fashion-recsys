"""A/B experiment power and simulation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_bundle, get_executor
from src.api.schemas import PowerRequest, PowerResult, SimulateRequest, SimulateResult
from src.api.services import experiment as exp_svc

router = APIRouter(tags=["experiment"])


@router.post(
    "/experiment/power",
    response_model=PowerResult,
    summary="Sample size and power calculation for an A/B test",
)
async def power(body: PowerRequest):
    return exp_svc.compute_power(
        baseline_rate=body.baseline_rate,
        relative_lift=body.relative_lift,
        alpha=body.alpha,
        power=body.power,
        weekly_traffic=body.weekly_traffic,
        cuped_variance_reduction=body.cuped_variance_reduction,
    )


@router.post(
    "/experiment/simulate",
    response_model=SimulateResult,
    summary="Run vectorized A/B or A/A simulation using week-104 per-customer arrays",
)
async def simulate(
    body: SimulateRequest, executor=Depends(get_executor), bundle=Depends(get_bundle)
):
    return await exp_svc.run_simulation(
        test_type=body.test_type,
        n_per_arm=body.n_per_arm,
        n_sims=body.n_sims,
        peeking=body.peeking,
        n_days=body.n_days,
        seed=body.seed,
        executor=executor,
        bundle=bundle,
    )
