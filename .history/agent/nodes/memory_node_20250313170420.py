from typing import Any, Dict
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from agent_state import AgentState
from langchain.chat_models import init_chat_model

buffer_memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_query", output_key="response")
summary_memory = ConversationSummaryMemory(llm=init_chat_model(model="gemma2-9b-it", model_provider="groq"), memory_key="summary_history", input_key="user_query", output_key="response")

def memory_node(state):
    """
        Memory node: Manages conversatuion history using Langchain's ConversationBufferMemory.
        Args: state (Dict[str, Any]): The current state dictionary containing user_query and chat_history.
        Returns: state (Dict[str, Any]): The updated state dictionary including chat_history.
    """
    user_query = state["user_query"]
    buffer_memory.chat_memory.messages = state.get("buffer_history", [])
    buffer_memory.chat_memory.add_message(user_query)
    updated_buffer_history = buffer_memory.chat_memory.messages
    
    current_summary = state.get("summary_history", [])
    summary_memory.buffer = current_summary
    updated_summary_history = summary_memory.predict_new_summary(current_summary, user_query)
    
    hybrid_context = "Detailed Recent Conversation:\n" + + "".join([str(m) for m in updated_buffer_history[-3:]]) if updated_buffer_history else ""
    hybrid_context += "\n\n\nSummary of Recent Conversation:\n" + updated_summary_history if updated_summary_history else ""
    
    updated_state = AgentState(
        user_query=user_query,
        buffer_history=updated_buffer_history,
        summary_history=updated_summary_history,
        conversation_history=hybrid_context,
        response=None
    )
    
    print("Memory Node State (Updated):", updated_state)
    return updated_state