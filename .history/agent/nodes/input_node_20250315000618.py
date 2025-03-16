from agent_state import AgentState

def input_node(state):
    """
        Input node: Receives the user query and initialises the state.
        Args: state (Dict[str, Any]): The current state dictionary.
        Returns: state (Dict[str, Any]): The updated state dictionary.
    """
    print("--- Input Node ---")
    user_query = state["user_query"]
    
    if not user_query:
        raise ValueError("User query not found in state.")
    
    print("Received user query:\n", user_query)
    
    agent_state: AgentState = {
        "user_query": user_query,
        "conversation_history": state.get("conversation_history", []),
    }
    
    print("Input Node state: \n", agent_state)
    return agent_state
