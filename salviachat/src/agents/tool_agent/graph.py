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
    """Create the bioinformatics computation and tool-call subgraph"""
    
    # 1. Instantiate the component that binds tools to the model.
    tool_component = ToolComponent()
    
    # 2. Create the graph.
    workflow = StateGraph(AgentState)
    
    # 3. Add nodes.
    # Node 1: plan tool calls, run Python tools, and summarize results.
    workflow.add_node("execute_tools", tool_component.execute_tools)
    
    # 4. Set the entry point.
    workflow.set_entry_point("execute_tools")
    
    workflow.add_edge("execute_tools", END)
    
    # Compile and return the subgraph.
    return workflow.compile()