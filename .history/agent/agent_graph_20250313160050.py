from langchain_core.runnables import RunnablePassthrough
from nodes import input_node, memory_node, output_node
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel

class AgentState(BaseModel):
    keys: dict  # Let's assume 'keys' is meant to be a dictionary for now

def create_agent_graph():
    """Create a graph for the agent."""
    builder = StateGraph(AgentState)
    
    builder.add_node("input_node", input_node.input_node)
    builder.add_node("memory_node", memory_node.memory_node)
    builder.add_node("output_node", output_node.output_node)
    
    builder.add_edge(START, "input_node")
    builder.add_edge("input_node", "memory_node")
    builder.add_edge("memory_node", "output_node")
    builder.add_edge("output_node", END)
    
    graph = builder.compile()
    return graph

if __name__ == "__main__":
    graph = create_agent_graph()
    print("Langgraph agent successfully created.")
    graph.invoke({"keys": {}})  # Assuming an empty dictionary for now