import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage

# 这个函数负责把新消息追加到历史消息列表中
def add_messages(left: list, right: list):
    return left + right

class AgentState(TypedDict):
    """
    这是 LangGraph 的全局状态定义。
    任何你想在 Agent 之间传递的数据，都要在这里声明。
    """
    # 1. 必须有的：对话历史记录
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 2. 传递额外的元数据 (比如当前执行到了哪一步，或者存放临时变量)
    metadata: Dict[str, Any]
    
    # 3. [新增] 专门用来在 salvia_agent 和 plot_agent 之间传递内存数据的字段
    extracted_data: List[Dict[str, Any]]