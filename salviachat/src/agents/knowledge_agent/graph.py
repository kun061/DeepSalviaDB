from langgraph.graph import StateGraph, END
from .component import KnowledgeComponent
# from typing import Annotated, Sequence
# from typing_extensions import TypedDict
# from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     metadata: dict
from ...core.state import AgentState
def create_knowledge_subgraph():
    """创建鼠尾草百科与文献问答子图"""
    
    # 1. 实例化组件
    knowledge_component = KnowledgeComponent()
    
    # 2. 创建图
    workflow = StateGraph(AgentState)
    
    # 3. 添加节点
    # 节点1: 检索向量库并回答普适性问题
    workflow.add_node("answer_question", knowledge_component.answer_general_question)
    
    # 4. 设置入口点
    workflow.set_entry_point("answer_question")
    
    # 5. 添加边
    # 因为百科问答通常是一步到位，不需要像查库那样核对 SQL，所以直接走向 END
    workflow.add_edge("answer_question", END)
    
    # 编译并返回子图
    return workflow.compile()