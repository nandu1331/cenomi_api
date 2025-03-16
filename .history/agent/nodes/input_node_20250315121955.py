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
    conversation_history: List[Dict[str, str]] = state.get("conversation_history", [])
    collected_offer_data: Dict[str, Any] = state.get("collected_offer_data", {})
    awaiting_tenant_input_field = state.get("awaiting_tenant_input_field")
    
    print(f"Input Node - User Query: {user_query}, Awaiting Input Field: {awaiting_tenant_input_field}")
    
    if not user_query:
        raise ValueError("User query not found in state.")
    
    print("Received user query:\n", user_query)
    
    # agent_state: AgentState = {
    #     "user_query": user_query,
    #     "conversation_history": state.get("conversation_history", []),
    # }
    
    updated_state: AgentState = state.copy()
    if awaiting_tenant_input_field: # If agent is waiting for specific input (multi-turn tenant action)
        print(f"Input Node - Agent is awaiting input for field: {awaiting_tenant_input_field}")
        collected_offer_data[awaiting_tenant_input_field] = user_query # Store tenant input in collected_offer_data
        updated_state["collected_offer_data"] = collected_offer_data # Update collected data in state
        updated_state["awaiting_tenant_input_field"] = None # Clear awaiting field as input is received
        updated_state["next_node"] = "tenant_action_node" # Route back to tenant action node to continue processing

    else: # Normal input (not part of multi-turn tenant action)
        print("Input Node - Normal user query (not awaiting specific input)")
        updated_state["user_query"] = user_query # Set user query in state
        updated_state["conversation_history"] = conversation_history # Pass conversation history (might already be there)
        updated_state["next_node"] = "memory_node"
    
    print("Input Node state: \n", updated_state)
    return updated_state
