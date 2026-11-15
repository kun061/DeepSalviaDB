import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage

def add_messages(left: list, right: list):
    return left + right

class AgentState(TypedDict):
    """
    Global LangGraph state.
    Declare here any data shared between agents.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    
    metadata: Dict[str, Any]
    
    extracted_data: List[Dict[str, Any]]
