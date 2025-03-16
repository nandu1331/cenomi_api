from nodes import input_node, memory_node, output_node
from langgraph.graph import StateGraph, END, START
from typing import Any
from agent_state import AgentState

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
    
    while True:
        user_query = input("Enter your query: ")
    
        input_data = {"user_query": user_query}
        result = graph.invoke(input_data)
        print("Result: \n", result)