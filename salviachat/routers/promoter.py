"""Proxy ProMoAI requests from the public gateway to the internal worker."""

import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
WORKER_URL = os.getenv("PROMOAI_WORKER_URL", "http://127.0.0.1:8003/predict")
HEALTH_URL = os.getenv("PROMOAI_HEALTH_URL", "http://127.0.0.1:8003/health")
WORKER_TIMEOUT = float(os.getenv("PROMOAI_WORKER_TIMEOUT", "120"))


class DnaRequest(BaseModel):
    sequence: str
    threshold: float = 0.5
    strand: str = "forward"


def response_detail(response: httpx.Response):
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


@router.get("/health")
async def promoter_health():
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(HEALTH_URL)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"ProMoAI worker unavailable: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=response_detail(exc.response),
        ) from exc


@router.post("/predict")
async def predict_promoter(req: DnaRequest):
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    try:
        async with httpx.AsyncClient(timeout=WORKER_TIMEOUT) as client:
            response = await client.post(WORKER_URL, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="ProMoAI worker 未启动；请先在 promoai 目录启动 8003 端口。",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="ProMoAI worker 推理超时；可调大 PROMOAI_WORKER_TIMEOUT。",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=response_detail(exc.response),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"ProMoAI worker unavailable: {exc}") from exc
