from typing import Dict, Any
from agent_state import AgentState
from langgraph.graph import END
from langchain.memory import ConversationBufferMemory

def output_node(state: Dict[str, Any]) -> AgentState:
    """Output node: Displays the result and prompts for a new query."""
    print("--- Output Node ---")
    
    response = state.get("response", "No response generated.")
    print(f"Agent Response: {response}")
    
    # Prompt for a new query
    new_query = input("Enter your next query (or type 'exit' to stop): ").strip()
    
    if new_query.lower() == "exit":
        return AgentState(
            user_query=state["user_query"],
            buffer_history=state["buffer_history"],
            summary_history=state["summary_history"],
            intent=state["intent"],
            selected_tools=state["selected_tools"],
            tool_outputs=state["tool_outputs"],
            response=response,
            next_node=END
        )
    else:
        return AgentState(
            user_query=new_query,
            buffer_history = state.get("buffer_history", ""),
            summary_history=state["summary_history"],
            intent=None,  # Reset for new intent classification
            selected_tools=[],
            tool_outputs={},
            response=None,
            next_node="input_node"  # Loop back to process the new query
        )