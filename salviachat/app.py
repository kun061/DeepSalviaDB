import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env", override=False)
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

app = FastAPI(title="Salvia Unified API Service", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        await init_rag_system()
        await warm_ollama_models()
        app.state.rag_status = "ok"
    except Exception as exc:
        app.state.rag_status = f"unavailable: {exc}"
        logger.exception("RAG initialization failed; other inference APIs remain available")


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
