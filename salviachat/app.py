import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env", override=False)
# print("=== LangSmith 环境变量检查 ===")
# print("TRACING:", os.environ.get("LANGCHAIN_TRACING_V2"))
# print("PROJECT:", os.environ.get("LANGCHAIN_PROJECT"))
# print("API_KEY:", os.environ.get("LANGCHAIN_API_KEY")[:8] + "..." if os.environ.get("LANGCHAIN_API_KEY") else "None")
# print("==============================")
from routers.chat import router as chat_router, init_rag_system
from routers.promoter import router as promoter_router
from routers.metabolome import router as metabolome_router
from routers.structure import router as structure_router
from routers.deepppi import router as deepppi_router
from routers.primer import router as primer_router
import logging
import httpx

logger = logging.getLogger(__name__)


async def warm_ollama_models():
    """Preload the two models so the first user does not pay cold-start latency."""
    enabled = os.getenv("SALVIACHAT_WARM_MODELS", "true").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    base_url = os.getenv("SALVIACHAT_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    models = (
        os.getenv("SALVIACHAT_SQL_MODEL", "qwen2.5-coder:7b"),
        os.getenv("SALVIACHAT_GENERATOR_MODEL", "qwen2.5:14b"),
    )
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            for model_name in dict.fromkeys(models):
                response = await client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": os.getenv("SALVIACHAT_KEEP_ALIVE", "24h"),
                    },
                )
                response.raise_for_status()
                logger.info("Ollama model warmed: %s", model_name)
    except Exception:
        logger.exception("Ollama warm-up failed; lazy loading will still be used")

# 1. 初始化主 App
app = FastAPI(title="Salvia Unified API Service", version="2.0")
# output_path = os.path.abspath("./outputs")
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
# app.mount("/static", StaticFiles(directory=output_path), name="static")
# 2. 全局配置跨域 (只写一次即可，所有子路由都生效)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 注册启动事件 (专门用于初始化 RAG 系统)
# 预测模型的初始化在 import routers.promoter 时就已经自动执行了
@app.on_event("startup")
async def startup_event():
    try:
        await init_rag_system()
        await warm_ollama_models()
        app.state.rag_status = "ok"
    except Exception as exc:
        app.state.rag_status = f"unavailable: {exc}"
        logger.exception("RAG initialization failed; other inference APIs remain available")

# 4. 挂载路由 (核心步骤)
# 这样设置后，接口地址会变成：
# 多Agent项目: http://IP:6006/rag/chat
# 预测项目:     http://IP:6006/pred/predict

app.include_router(chat_router, prefix="/rag", tags=["Multi-Agent RAG"])
app.include_router(promoter_router, prefix="/pred", tags=["Promoter Prediction"])
app.include_router(metabolome_router, prefix="/metabolome", tags=["Metabolome Prediction"])
app.include_router(structure_router, prefix="/structure", tags=["Protein 3D Structure"])
app.include_router(deepppi_router, prefix="/ppi", tags=["Protein Interaction"])
app.include_router(primer_router, prefix="/primer", tags=["Primer Design Engine"])

@app.get("/health", tags=["Service"])
def health():
    from routers.promoter import model, model_load_error
    return {
        "status": "ok",
        "services": {
            "salviachat": getattr(app.state, "rag_status", "initializing"),
            "promoai": "ok" if model is not None else f"unavailable: {model_load_error}",
            "metabolome": "proxied via METABOLOME_WORKER_URL",
            "deepppi": "proxied via DEEPPPI_WORKER_URL",
        },
        "docs": "/docs",
    }

if __name__ == "__main__":
    # 本地直接运行入口。
    uvicorn.run("app:app", host="0.0.0.0", port=6006, reload=False)