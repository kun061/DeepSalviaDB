from langgraph.graph import StateGraph, END
from .component import ToolComponent
# from typing import Annotated, Sequence
# from typing_extensions import TypedDict
# from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     metadata: dict
from ...core.state import AgentState
def create_tool_subgraph():
    """创建生物信息学动态计算与工具调用子图"""
    
    # 1. 实例化核心组件 (大模型绑定工具的逻辑都在这里面)
    tool_component = ToolComponent()
    
    # 2. 创建图
    workflow = StateGraph(AgentState)
    
    # 3. 添加节点
    # 节点1: 执行大模型工具调度逻辑 (包含思考、调用Python脚本、汇总结果)
    workflow.add_node("execute_tools", tool_component.execute_tools)
    
    # 4. 设置入口点
    workflow.set_entry_point("execute_tools")
    
    workflow.add_edge("execute_tools", END)
    
    # 编译并返回子图
    return workflow.compile()