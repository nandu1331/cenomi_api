from agent.nodes import input_node, memory_node, output_node, intent_router_node, tool_selection_node, tool_invocation_node, llm_call_node, tenant_action_node
from langgraph.graph import StateGraph, END, START
from typing import Any
from agent.agent_state import AgentState

def create_agent_graph():
    """Create a graph for the agent."""
    builder = StateGraph(AgentState)
    
    builder.add_node("input_node", input_node.input_node)
    builder.add_node("memory_node", memory_node.memory_node)
    builder.add_node("intent_router_node", intent_router_node.intent_router_node)
    builder.add_node("tool_selection_node", tool_selection_node.tool_selection_node)
    builder.add_node("tool_invocation_node", tool_invocation_node.tool_invocation_node)
    builder.add_node("llm_call_node", llm_call_node.llm_call_node)
    builder.add_node("output_node", output_node.output_node)
    builder.add_node("tenant_action_node", tenant_action_node.tenant_action_node)
    
    def route_from_input(state: AgentState):
        if state.get("awaiting_tenant_input_field"):
            print("Input Router: Awaiting tenant input - routing back to tenant_action_node")
            return "tenant_action_node"
        else:
            print("Input Router: Normal user query - routing to intent_router_node")
            return "intent_router_node"
    
    builder.add_edge(START, "input_node")
    builder.add_conditional_edges("input_node", route_from_input, {
        "tenant_action_node": "tenant_action_node",
        "intent_router_node": "intent_router_node"
    })
    
    def should_continue(state):
        if state.get("next_node") == "tool_selection_node":
            return "tool_selection_node"
        elif state.get("next_node") == "tenant_action_node":
            return "tenant_action_node"
        elif state.get("next_node") == "llm_call_node":
            return "llm_call_node"
        elif state.get("next_node") == "output_node":
            return "output_node"
        else:
            return "output_node"
    
    builder.add_conditional_edges("intent_router_node", should_continue)
    builder.add_conditional_edges("tenant_action_node", should_continue)
    builder.add_edge("tool_selection_node", "tool_invocation_node")
    builder.add_edge("tool_invocation_node", "llm_call_node")
    builder.add_edge("llm_call_node", "memory_node")
    builder.add_edge("memory_node", "output_node")
    
    def route_from_output(state: AgentState) -> str:
        return state["next_node"] if state["next_node"] else END
    
    builder.add_conditional_edges("output_node", route_from_output)
    
    graph = builder.compile()
    return graph