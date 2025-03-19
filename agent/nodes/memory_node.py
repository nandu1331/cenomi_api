from agent.agent_state import AgentState

def memory_node(state: AgentState) -> AgentState:
    if "conversation_history" not in state or state["conversation_history"] is None:
        state["conversation_history"] = []
    
    user_query = state.get("user_query", "")
    assistant_response = state.get("response", "").strip()
    
    if assistant_response and assistant_response != "Sorry, I couldn't process your request.":
        state["conversation_history"].append({"user": user_query, "bot": assistant_response})
    
    updated_state = state.copy()
    return updated_state