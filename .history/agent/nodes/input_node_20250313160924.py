def input_node(state):
    """
        Input node: Receives the user query and initialises the state.
        Args: state (Dict[str, Any]): The current state dictionary.
        Returns: state (Dict[str, Any]): The updated state dictionary.
    """
    user_query = state.get("user_query")
    
    if not user_query:
        raise ValueError("User query not found in state.")
    
    print("Received user query:", user_query)
    return {"user_query": user_query}
