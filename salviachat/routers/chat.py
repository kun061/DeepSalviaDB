import logging
import time
import json
import re
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse # 导入流式响应
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.core.settings import configure_settings
from src.core.rag import initialize_rag_system
logger = logging.getLogger(__name__)
# --- 数据模型定义 (保持原样) ---
class QueryRequest(BaseModel):
    question: str
class TableColumn(BaseModel):
    prop: str
    label: str
class TableData(BaseModel):
    columns: List[TableColumn]
    data: List[Dict[str, Any]]
class ChartData(BaseModel):
    type: str
    title: str
    labels: List[str]
    values: List[float]
class ChatResponse(BaseModel):
    question: str
    answer: str
    table: Optional[TableData] = None
    chart: Optional[ChartData] = None
    trace: List[Dict[str, Any]]
router = APIRouter()
rag_workflow = None
async def init_rag_system():
    global rag_workflow
    logger.info("Initializing RAG System...")
    configure_settings()
    rag_workflow = initialize_rag_system()
    logger.info("RAG System Ready.")

@router.post("/chat")
async def chat_endpoint(request: Request, body: QueryRequest):
    if not rag_workflow:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    async def stream_generator(req: Request):
        inputs = {"messages": [HumanMessage(content=body.question, additional_kwargs={"type": "user_question"})]}
        started_at = time.perf_counter()
        last_node = None
        yield f"data: {json.dumps({'type': 'status', 'stage': 'accepted'}, ensure_ascii=False)}\n\n"
        
        try:
            async for msg_chunk, metadata in rag_workflow.astream(inputs, stream_mode="messages"):
                node_name = metadata.get("langgraph_node", "")
                if node_name and node_name != last_node:
                    last_node = node_name
                    status_payload = {
                        "type": "status",
                        "stage": node_name,
                        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                    }
                    yield f"data: {json.dumps(status_payload, ensure_ascii=False)}\n\n"
                
                if await req.is_disconnected():
                    logger.info(f"前端主动断开连接，立即终止大模型推理。问题: {body.question}")
                    break  # 直接跳出循环，终止 LangGraph 运行
                
                if node_name == "Generator" or "generat" in node_name.lower() or node_name == "rag_node":
                    
                    if type(msg_chunk).__name__ == "AIMessageChunk":
                        if msg_chunk.content:
                            payload = {"content": msg_chunk.content}
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    
                    elif type(msg_chunk).__name__ == "AIMessage":
                        if "suggest" in node_name.lower() and msg_chunk.content:
                            payload = {"content": msg_chunk.content}
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        if hasattr(msg_chunk, "additional_kwargs") and msg_chunk.additional_kwargs:
                            kwargs = msg_chunk.additional_kwargs
                            extra_payload = {}
                            
                            if kwargs.get("chart_data"):
                                extra_payload["chart"] = kwargs["chart_data"]
                                
                            raw_table = kwargs.get("table_data")
                            if raw_table and isinstance(raw_table, list) and len(raw_table) > 0:
                                keys = raw_table[0].keys()
                                header_map = {
                                    "gene_id": "Gene ID", "species_name": "Species", 
                                    "start_bp": "Start", "end_bp": "End", 
                                    "fpkm": "FPKM", "rpkm": "RPKM"
                                }
                                columns = [{"prop": k, "label": header_map.get(k, k)} for k in keys]
                                extra_payload["table"] = {"columns": columns, "data": raw_table}

                            if extra_payload:
                                yield f"data: {json.dumps(extra_payload, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error processing stream request: {error_details}")
            error_payload = {"content": f"\n\n❌ Internal Bug: {str(e)}"}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" 
        }
    )
