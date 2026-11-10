"""Proxy Metabolome requests from the public gateway to the internal worker."""

import os
from typing import List

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
WORKER_URL = os.getenv(
    "METABOLOME_WORKER_URL",
    "http://127.0.0.1:8002/predict_multi_gene",
)
HEALTH_URL = os.getenv(
    "METABOLOME_HEALTH_URL",
    "http://127.0.0.1:8002/health",
)
WORKER_TIMEOUT = float(os.getenv("METABOLOME_WORKER_TIMEOUT", "600"))


class PredictionRequest(BaseModel):
    metabolite: str
    stage: str
    control_stage: str = "DAG30"
    model_types: List[str] = Field(default_factory=lambda: ["RandomForest"])
    top_n: int = 20


def response_detail(response: httpx.Response):
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


@router.get("/health")
async def metabolome_health():
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(HEALTH_URL)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Metabolome worker unavailable: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=response_detail(exc.response),
        ) from exc


@router.post("/predict_multi_gene")
async def predict_metabolome_genes(req: PredictionRequest):
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    try:
        async with httpx.AsyncClient(timeout=WORKER_TIMEOUT) as client:
            response = await client.post(WORKER_URL, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Metabolome worker 未启动；请先在 metabolome_service 目录启动 8002 端口。",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Metabolome worker 计算超时；可调大 METABOLOME_WORKER_TIMEOUT。",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=response_detail(exc.response),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Metabolome worker unavailable: {exc}") from exc
