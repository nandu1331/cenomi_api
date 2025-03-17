from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from agent.agent_state import AgentState
from config.config_loader import load_config
from langchain_google_genai import ChatGoogleGenerativeAI

config = load_config()

buffer_memory = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key="user_query", 
    output_key="response"
)
summary_memory = ConversationSummaryMemory(
    
    llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key),
    memory_key="summary_history", 
    input_key="user_query", 
    output_key="response"
)

def memory_node(state: AgentState) -> AgentState:
    """
    Memory node: Manages conversation history using Langchain's memory objects.
    It now saves only the current turn.
    """
    print("--- Memory Node ---")
    user_query = state["user_query"]
    assistant_response = state.get("response", "").strip()
    tenant_main_query = state.get("tenant_main_query", "")
    
    # Save only the current turn
    user_queries = {"user_query": user_query, "tenant_main_query": tenant_main_query}
    buffer_memory.save_context(user_queries, {"response": assistant_response})
    summary_memory.save_context(user_queries, {"response": assistant_response})
    
    # Load the updated histories from memory
    buffer_history_dict = buffer_memory.load_memory_variables({})
    updated_buffer_history = buffer_history_dict.get("chat_history", "")
    
    summary_history_dict = summary_memory.load_memory_variables({})
    updated_summary_history = summary_history_dict.get("summary_history", "")
    
    # Build a hybrid context for downstream use
    hybrid_context = "Detailed Recent Conversation:\n" + updated_buffer_history
    hybrid_context += "\n\nSummary of Recent Conversation:\n" + updated_summary_history
    
    updated_state = AgentState(
        user_query=user_query,
        buffer_history=updated_buffer_history,
        summary_history=updated_summary_history,
        conversation_history=hybrid_context,
        response=state["response"]  # Reset response for new input
    )
    
    print("Memory Node State (Updated):", updated_state)
    return updated_state
