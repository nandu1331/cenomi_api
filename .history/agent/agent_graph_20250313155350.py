from langchain_core.runnables import RunnablePassthrough
from nodes import input_node, memory_node, output_node
from langgraph.graph import StateGraph, END

def create_agent_graph():
    """Create a graph for the agent."""
    builder = StateGraph(lambda state: {"keys": state.keys()})
    
    builder.add_node("input_node", input_node.input_node)
    builder.add_node("memory_node", memory_node.memory_node)
    builder.add_node("output_node", output_node.output_node)
    
    builder.set_entry_point("input_node")
    builder.add_edge("input_node", "memory_node")
    builder.add_edge("memory_node", "output_node")
    builder.add_edge("output_node", END)
    
    graph = builder.compile()
    return graph

if __name__ == "__main__":
    graph = create_agent_graph()
    print("Langgraph agent successfully created.")
    graph.run()