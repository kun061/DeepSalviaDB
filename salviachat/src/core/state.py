import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage

def add_messages(left: list, right: list):
    return left + right

class AgentState(TypedDict):
    """
    这是 LangGraph 的全局状态定义。
    任何你想在 Agent 之间传递的数据，都要在这里声明。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    
    metadata: Dict[str, Any]
    
    extracted_data: List[Dict[str, Any]]
