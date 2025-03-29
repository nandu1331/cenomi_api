from agent.agent_state import AgentState
from agent.utils.database_utils import db_execute

def memory_node(state: AgentState) -> AgentState:
    conversation_history = state.get("conversation_history", [])
    if isinstance(conversation_history, str):
        conversation_history = []

    user_query = state["user_query"]
    assistant_response = state.get("response", "").strip()
    
    new_entry = {
        "user_query": user_query,
        "response": assistant_response
    }

    updated_conversation_history = (conversation_history + [new_entry])[-10:]
    hybrid_context = "Detailed Recent Conversation:\n" + "\n".join(
        [f"User: {entry['user_query']}\nAssistant: {entry['response']}" 
         for entry in updated_conversation_history]
    )
    
    updated_state = state.copy()
    updated_state["conversation_history"] = updated_conversation_history
    updated_state["conversation_history_str"] = hybrid_context 

    # Insert messages into conversation_messages
    session_id = state["session_id"]
    # User message
    db_execute(
        """
        INSERT INTO conversation_messages (session_id, role, content, timestamp)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        """,
        (session_id, "user", user_query)
    )
    # Assistant message
    db_execute(
        """
        INSERT INTO conversation_messages (session_id, role, content, timestamp)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        """,
        (session_id, "assistant", assistant_response)
    )

    return updated_state