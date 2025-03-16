from typing import Any, Dict
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from agent_state import AgentState
from langchain.chat_models import init_chat_model

buffer_memory = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key="user_query", 
    output_key="response"
)
summary_memory = ConversationSummaryMemory(
    llm=init_chat_model(model="gemma2-9b-it", model_provider="groq"), 
    memory_key="summary_history", 
    input_key="user_query", 
    output_key="response"
)

def memory_node(state):
    """
        Memory node: Manages conversatuion history using Langchain's ConversationBufferMemory.
        Args: state (Dict[str, Any]): The current state dictionary containing user_query and chat_history.
        Returns: state (Dict[str, Any]): The updated state dictionary including chat_history.
    """
    user_query = state["user_query"]
    prior_buffer_memory = state.get("buffer_history", [])
    if prior_buffer_memory:
        for msg in prior_buffer_memory:
            buffer_memory.save_context({"user_query": str(msg)}, {"response": ""})
            
    buffer_memory.save_context({"user_query": user_query}, {"response": ""})
    buffer_history_dict = buffer_memory.load_memory_variables({})
    updated_buffer_history = buffer_history_dict["chat_history"]
    
    prior_summary_memory = state.get("summary_history", "")
    if prior_summary_memory:
        summary_memory.buffer = prior_summary_memory
        
    summary_memory.save_context({"user_query": user_query}, {"response": ""})
    summary_history_dict = summary_memory.load_memory_variables({})
    updated_summary_history = summary_history_dict["summary_history"]
    
    hybrid_context = "Detailed Recent Conversation:\n"
    if updated_buffer_history:
        recent_messages = updated_buffer_history[-5:]
        hybrid_context += "\n".join([str(m) for m in recent_messages])
    hybrid_context += "\n\nSummary of Recent Conversation:\n" + updated_summary_history
    
    updated_state = AgentState(
        user_query=user_query,
        buffer_history=updated_buffer_history,
        summary_history=updated_summary_history,
        conversation_history=hybrid_context,
        response=None
    )
    
    print("Memory Node State (Updated):", updated_state)
    return updated_state