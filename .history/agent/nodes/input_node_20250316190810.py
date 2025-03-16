from agent_state import AgentState
from typing import List, Dict, Any

def input_node(state):
    """
        Input node: Receives the user query and initialises the state.
        Args: state (Dict[str, Any]): The current state dictionary.
        Returns: state (Dict[str, Any]): The updated state dictionary.
    """
    print("--- Input Node ---")
    user_query = state["user_query"]
    user_query = state.get("user_query")
    conversation_history: List[Dict[str, str]] = state.get("conversation_history", [])
    tenant_data: Dict[str, Any] = state.get("tenant_data", {})
    awaiting_tenant_input_field = state.get("awaiting_tenant_input_field")
    if not current_field_index:
        current_field_index = 0
    current_field_index: int = state.get("current_field_index", 0)
    
    print(f"Input Node - User Query: {user_query}, Awaiting Input Field: {awaiting_tenant_input_field}")

    updated_state: AgentState = state.copy()

    if awaiting_tenant_input_field: # If agent is waiting for specific input (multi-turn tenant action)
        print(f"Input Node - Agent is awaiting input for field: {awaiting_tenant_input_field}")
        tenant_data[awaiting_tenant_input_field] = user_query # Store tenant input in tenant_data - CORRECTED LINE
        updated_state["tenant_data"] = tenant_data # Update tenant_data in state - CORRECTED LINE
        # updated_state["awaiting_tenant_input_field"] = None # Clear awaiting field as input is received 
        
        current_field_index += 1
        updated_state["current_field_index"]
        
        updated_state["next_node"] = "tenant_action_node"

    else: # Normal input (not part of multi-turn tenant action)
        print("Input Node - Normal user query (not awaiting specific input)")
        updated_state["user_query"] = user_query # Set user query in state
        updated_state["conversation_history"] = conversation_history # Pass conversation history (might already be there)
        updated_state["next_node"] = "intent_router_node" # Proceed to memory node for intent routing

    print("Input Node State (Updated):", updated_state)
    return updated_state