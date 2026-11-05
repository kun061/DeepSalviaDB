import os
from dotenv import load_dotenv

def configure_settings():
    load_dotenv()
    # os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
    # os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    # os.environ["LANGSMITH_PROJECT_NAME"] = os.getenv("LANGSMITH_PROJECT_NAME", "OpenBioLLM-RAG")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USER_AGENT"] = "OpenBioLLM-RAG/1.0"

    # --- 新增：Neo4j 图数据库配置 ---
    # os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    # os.environ["NEO4J_USER"] = os.getenv("NEO4J_USER", "neo4j")
    # os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD", "wk20000601")
