"""Consistent error envelope for all API errors."""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("api")


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str = ""


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


def make_error(code: str, message: str, request_id: str = "") -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


async def http_exception_handler(request: Request, exc) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=exc.status_code,
        content=make_error(f"HTTP_{exc.status_code}", str(exc.detail), request_id),
    )


async def validation_exception_handler(request: Request, exc) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=422,
        content=make_error("VALIDATION_ERROR", str(exc), request_id),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=make_error("INTERNAL_ERROR", "An unexpected error occurred.", request_id),
    )
