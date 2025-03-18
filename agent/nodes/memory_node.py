from agent.agent_state import AgentState

def memory_node(state: AgentState) -> AgentState:
    """
    Memory Node: Updates conversation history within the graph for the current invocation.
    """
    print("--- Memory Node ---")
    
    # Ensure conversation_history exists
    if "conversation_history" not in state or state["conversation_history"] is None:
        state["conversation_history"] = []
    
    user_query = state.get("user_query", "")
    assistant_response = state.get("response", "").strip()
    
    # Append only if there's a valid response (avoid duplicates or errors)
    if assistant_response and assistant_response != "Sorry, I couldn't process your request.":
        state["conversation_history"].append({
            "user": user_query,
            "bot": assistant_response
        })
    
    updated_state = state.copy()
    print("Memory Node State (Updated):", updated_state)
    return updated_state