import os
from dotenv import load_dotenv

def configure_settings():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USER_AGENT"] = "OpenBioLLM-RAG/1.0"
