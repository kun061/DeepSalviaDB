# DeepPPI proxy router used by salviachat/app.py.
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx  # 使用 httpx 进行跨服务调用

router = APIRouter()

class PPIRequest(BaseModel):
    seq_a: str
    seq_b: str
    model_type: str = "ensemble"
    threshold: float = 0.5
# 指向独立 DeepPPI worker 的地址
WORKER_URL = os.getenv("DEEPPPI_WORKER_URL", "http://127.0.0.1:8001/predict_ppi")

@router.post("/predict_ppi")
async def predict_ppi(req: PPIRequest):
    try:
        # 使用 httpx 异步将请求转发给 8001 端口的 DeepPPI worker
        async with httpx.AsyncClient() as client:
            payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            response = await client.post(WORKER_URL, json=payload, timeout=120.0)
            response.raise_for_status() # 检查 HTTP 错误
            
            # 获取 worker 算出来的结果
            result = response.json()
            
        return {
            "status": "success",
            "model": result.get("model_used", "Unknown Model"),
            "probability": round(result["probability"], 4),
            "interaction": result["interaction"],
            "importance_track_a": result.get("importance_track_a", []),
            "importance_track_b": result.get("importance_track_b", []),
            "threshold": result.get("threshold", req.threshold),
            "probabilities_by_seed": result.get("probabilities_by_seed", {}),
            "sequence_lengths": result.get("sequence_lengths", {}),
            "truncated_to": result.get("truncated_to", 512),
        }
        
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503, 
            detail="DeepPPI Worker 未启动，请在 salvia-deepppi-d 环境运行 api_service.py"
        )
    except httpx.HTTPStatusError as e:
        detail = e.response.text
        try:
            detail = e.response.json().get("detail", detail)
        except ValueError:
            pass
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))