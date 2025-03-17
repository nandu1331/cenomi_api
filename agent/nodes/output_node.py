from typing import Dict, Any
from agent.agent_state import AgentState
from langgraph.graph import END

def output_node(state: Dict[str, Any]) -> AgentState:
    """Output node: Prepares the final response."""
    print("--- Output Node ---")
    
    response = state.get("response", "No response generated.")
    awaiting_tenant_input_field = state.get("awaiting_tenant_input_field")
    print(f"Agent Response: {response}")
    
    updated_state: AgentState = state.copy()
    updated_state["agent_response"] = response
    
    if awaiting_tenant_input_field:
        updated_state["next_node"] = "input_node"  # Route back for tenant input
    else:
        updated_state["next_node"] = END  # End the flow for API response
    
    print("Output Node State (Updated):", updated_state)
    return updated_state