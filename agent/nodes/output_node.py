from typing import Dict, Any
from agent.agent_state import AgentState
from langgraph.graph import END

def output_node(state: Dict[str, Any]) -> AgentState:
    response = state.get("response", "No response generated.")
    awaiting_tenant_input_field = state.get("awaiting_tenant_input_field")
    
    updated_state: AgentState = state.copy()
    updated_state["agent_response"] = response
    
    updated_state["next_node"] = "input_node" if awaiting_tenant_input_field else END
    
    return updated_state