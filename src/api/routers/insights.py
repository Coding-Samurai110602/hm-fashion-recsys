"""Insight endpoints backed by pre-computed report JSON files."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.services import insights as svc

router = APIRouter(tags=["insights"])


@router.get("/insights/summary", summary="Headline metrics including week-104 holdout")
async def summary():
    return svc.get_summary()


@router.get("/insights/weekly-lift", summary="Rolling-origin MAP@12 by week")
async def weekly_lift():
    return svc.get_weekly_lift()


@router.get("/insights/segments", summary="Segment-level MAP@12 analysis")
async def segments():
    return svc.get_segments()


@router.get("/insights/ablations", summary="Feature group ablation deltas")
async def ablations():
    return svc.get_ablations()


@router.get("/insights/importance", summary="SHAP importance vs gain importance")
async def importance():
    return svc.get_importance()
