from typing import Annotated, Sequence, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage
from .state import AgentState
from .router import Router
from .evaluator import Evaluator
from .generator import Generator
from ..agents.salvia_agent.graph import create_salvia_subgraph
from ..agents.tool_agent.graph import create_tool_subgraph

def suggest_node(state: AgentState):
    """
    当数据库明确查不到数据时，路由会流转到这里。
    你可以在这里用 LLM 动态生成推荐，或者先简单返回一段写死的兜底话术。
    """
    suggestion_text = "非常抱歉，目前的数据库中未能检索到该确切数据。建议您检查输入的基因ID或物种名称是否正确，或者尝试查询其他相关途径。"
    return {
        "messages": [AIMessage(content=suggestion_text)]
    }
def initialize_rag_system(disable_evaluator: bool = False):
    # Create components
    router = Router()
    evaluator = None if disable_evaluator else Evaluator()
    generator = Generator()
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router.route)
    if evaluator is not None:
        workflow.add_node("evaluator", evaluator.evaluate)
    workflow.add_node("generator", generator.generate)
    workflow.add_node("salvia_agent", create_salvia_subgraph())  
    workflow.add_node("tool_agent", create_tool_subgraph())
    # Add edges
    workflow.add_edge(START, "router")  # From START to router
    
    # Router's conditional routing
    workflow.add_conditional_edges(
        "router",
        lambda x: END if x.get("status") == "error" else x["next"],
        {
            "salvia_agent": "salvia_agent",
            "tool_agent" : "tool_agent",
            "irrelevant_questions": "generator",
        }
    )
    
    # Agent's conditional routing
    downstream = "generator" if disable_evaluator else "evaluator"
    for agent in ["salvia_agent", "tool_agent"]:
        workflow.add_conditional_edges(
            agent,
            lambda x: END if x.get("status") == "error" else downstream,
            {
                downstream: downstream,
            }
        )
    
    # Evaluator's conditional routing
    if not disable_evaluator:
        workflow.add_conditional_edges(
            "evaluator",
            lambda x: (
                END if x.get("status") == "error" 
                else x["next"]  # "generate" or "router"
            ),
            {
                "generate": "generator",
                "router": "router",
                "suggest": "generator",
            }
        )
    
    # Generator generates final answer
    workflow.add_edge("generator", END)
    
    # Compile workflow
    graph = workflow.compile()
    
    # Print Mermaid format graph
    print("\nMermaid Graph:")
    print("```mermaid")
    print(graph.get_graph().draw_mermaid())
    print("```")
    
    return graph
