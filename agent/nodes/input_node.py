from agent.agent_state import AgentState
from typing import List, Dict, Any

def input_node(state):
    """
        Input node: Receives the user query and initialises the state.
        Args: state (Dict[str, Any]): The current state dictionary.
        Returns: state (Dict[str, Any]): The updated state dictionary.
    """
    #("--- Input Node ---")
    user_query = state.get("user_query")
    conversation_history: List[Dict[str, str]] = state.get("conversation_history", [])
    role = state.get("role", "GUEST")  # Default to GUEST if not provided
    mall_name = state.get("mall_name", "")
    user_id = state.get("user_id")
    session_id = state.get("session_id")

    updated_state: AgentState = state.copy()
    updated_state["role"] = role
    updated_state["user_id"] = user_id
    updated_state["mall_name"] = mall_name
    updated_state["user_query"] = user_query 
    updated_state["conversation_history"] = conversation_history
    updated_state["session_id"] = session_id
    updated_state["tenant_data"] = {}
    updated_state["next_node"] = "intent_router_node"

    return updated_state