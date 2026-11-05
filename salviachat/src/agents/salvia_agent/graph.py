from langgraph.graph import StateGraph, END
from .component import SalviaComponent
# from typing import Annotated, Sequence
# from typing_extensions import TypedDict
# from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages


# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     metadata: dict
from ...core.state import AgentState
def create_salvia_subgraph():
    """创建鼠尾草数据库查询子图"""
    
    # 实例化组件
    salvia_component = SalviaComponent()
    
    # 创建图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    # 节点1: 生成 SQL
    workflow.add_node("generate_sql", salvia_component.generate_sql)
    # 节点2: 执行 SQL
    workflow.add_node("execute_sql", salvia_component.execute_sql)
    
    # 设置入口点
    workflow.set_entry_point("generate_sql")
    
    # 添加条件边 (生成成功则去执行，失败则结束)
    workflow.add_conditional_edges(
        "generate_sql",
        lambda x: x.get("next", END) if "next" in x else END,
        {
            "execute_sql": "execute_sql",
            END: END
        }
    )
    
    # 执行完 SQL 后结束，返回给主 Agent (Router/Generator)
    workflow.add_edge("execute_sql", END)
    
    return workflow.compile()