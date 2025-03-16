from typing import Any, Dict
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_query", output_key="response")

def memory_node(state):
    """
        Memory node: Manages conversatuion history using Langchain's ConversationBufferMemory.
        Args: state (Dict[str, Any]): The current state dictionary containing user_query and chat_history.
        Returns: state (Dict[str, Any]): The updated state dictionary including chat_history.
    """
    user_query = state.get("user_query")
    if not user_query:
        raise ValueError("User query not found in state.")
    
    memory.load_memory_variables(state)
    memory.save_context({"user_query": user_query, "response": "Acknowledged."})
    
    chat_history = memory.load_memory_variables({})["chat_history"]
    print("Chat history: \n", chat_history)
    
    updated_state = {"chat_history": chat_history, "user_query": user_query}
    return updated_state