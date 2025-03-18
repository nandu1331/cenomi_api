from langgraph.graph import StateGraph, END, START
from agent.nodes import input_node, intent_router_node, tool_selection_node, tool_invocation_node, llm_call_node, memory_node, output_node, tenant_action_node
from agent.agent_state import AgentState

def create_agent_graph():
    """Create a graph for the agent."""
    builder = StateGraph(AgentState)
    
    builder.add_node("input_node", input_node.input_node)
    builder.add_node("intent_router_node", intent_router_node.intent_router_node)
    builder.add_node("tool_selection_node", tool_selection_node.tool_selection_node)
    builder.add_node("tool_invocation_node", tool_invocation_node.tool_invocation_node)
    builder.add_node("llm_call_node", llm_call_node.llm_call_node)
    builder.add_node("memory_node", memory_node.memory_node)
    builder.add_node("output_node", output_node.output_node)
    builder.add_node("tenant_action_node", tenant_action_node.tenant_action_node)
    
    def route_from_input(state: AgentState) -> str:
        if state.get("awaiting_tenant_input_field"):
            print("Input Router: Awaiting tenant input - routing back to tenant_action_node")
            return "tenant_action_node"
        print("Input Router: Normal user query - routing to intent_router_node")
        return "intent_router_node"
    
    builder.add_edge(START, "input_node")
    builder.add_conditional_edges("input_node", route_from_input, {
        "tenant_action_node": "tenant_action_node",
        "intent_router_node": "intent_router_node"
    })
    
    def route_from_intent(state: AgentState) -> str:
        intent = state.get("intent")
        role = state.get("role", "anonymous")
        
        # Tenant-specific intents
        tenant_intents = [
            "TENANT_INSERT_STORE", "TENANT_UPDATE_STORE", "TENANT_DELETE_STORE",
            "TENANT_INSERT_OFFER", "TENANT_UPDATE_OFFER", "TENANT_DELETE_OFFER"
        ]
        
        if intent in tenant_intents:
            if role != "tenant":
                print("Intent Router: Non-tenant user attempted tenant action - routing to llm_call_node")
                return "llm_call_node"  # For a "permission denied" response
            print("Intent Router: Tenant intent detected - routing to tenant_action_node")
            return "tenant_action_node"
        
        print("Intent Router: Non-tenant intent - routing to tool_selection_node")
        return "tool_selection_node"
    
    builder.add_conditional_edges("intent_router_node", route_from_intent, {
        "tenant_action_node": "tenant_action_node",
        "tool_selection_node": "tool_selection_node",
        "llm_call_node": "llm_call_node"
    })
    
    def route_from_tenant_action(state: AgentState) -> str:
        if state.get("awaiting_tenant_input_field"):
            print("Tenant Action Router: Still awaiting input - looping back to input_node")
            return "input_node"
        print("Tenant Action Router: Action complete - routing to llm_call_node")
        return "llm_call_node"
    
    builder.add_conditional_edges("tenant_action_node", route_from_tenant_action, {
        "input_node": "input_node",
        "llm_call_node": "llm_call_node"
    })
    
    builder.add_edge("tool_selection_node", "tool_invocation_node")
    builder.add_edge("tool_invocation_node", "llm_call_node")
    builder.add_edge("llm_call_node", "memory_node")
    builder.add_edge("memory_node", "output_node")
    
    def route_from_output(state: AgentState) -> str:
        next_node = state.get("next_node", END)
        print(f"Output Router: Routing to {next_node}")
        return next_node
    
    builder.add_conditional_edges("output_node", route_from_output)
    
    graph = builder.compile()
    return graph