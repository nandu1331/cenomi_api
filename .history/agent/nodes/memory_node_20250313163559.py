from typing import Any, Dict
from langchain.memory import ConversationBufferMemory
from agent_graph import AgentState

memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_query", output_key="response")

def memory_node(state):
    """
        Memory node: Manages conversatuion history using Langchain's ConversationBufferMemory.
        Args: state (Dict[str, Any]): The current state dictionary containing user_query and chat_history.
        Returns: state (Dict[str, Any]): The updated state dictionary including chat_history.
    """
    user_query = state["user_query"]
    conversation_history = state.get("chat_history", [])
    print("Conversation history: \n", conversation_history)
    
    print("Current Conversation History:", conversation_history) # Print current history

    # --- Update Conversation History ---
    conversation_history.append({"user": user_query}) # Add user query to history
    print("Updated Conversation History:", conversation_history) # Print updated history

    # --- Update Agent State ---
    updated_state: AgentState = state.copy() # Create a copy to avoid modifying original state directly
    updated_state['conversation_history'] = conversation_history # Update history in the copied state

    print("Memory Node State:", updated_state) # Print state after memory update

    return updated_state