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
    """Create the Salvia database query subgraph"""
    
    # Instantiate the component.
    salvia_component = SalviaComponent()
    
    # Create the graph.
    workflow = StateGraph(AgentState)
    
    # Add nodes.
    # Node 1: generate SQL.
    workflow.add_node("generate_sql", salvia_component.generate_sql)
    # Node 2: execute SQL.
    workflow.add_node("execute_sql", salvia_component.execute_sql)
    
    # Set the entry point.
    workflow.set_entry_point("generate_sql")
    
    # Add a conditional edge: execute generated SQL or end.
    workflow.add_conditional_edges(
        "generate_sql",
        lambda x: x.get("next", END) if "next" in x else END,
        {
            "execute_sql": "execute_sql",
            END: END
        }
    )
    
    # Return to the main agent after executing SQL.
    workflow.add_edge("execute_sql", END)
    
    return workflow.compile()