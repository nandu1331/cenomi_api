from typing import Dict, Any
from agent_state import AgentState
from langgraph.graph import END

def output_node(state: Dict[str, Any]) -> AgentState:
    """Output node: Displays the result and prompts for a new query."""
    print("--- Output Node ---")
    
    response = state.get("response", "No response generated.")
    awaiting_tenant_input_field = state.get("awaiting_tenant_input_field")
    print(f"Agent Response: {response}")
    
    updated_state: AgentState = state.copy()
    
    if awaiting_tenant_input_field:
        next_node = "input_node" # Route to input node to get tenant input
    else:
        next_node = "llm_call_node"
    
    # Prompt for a new query
    new_query = input("Enter your next query (or type 'exit' to stop): ").strip()
    
    updated_state["user_query"] = new_query
    updated_state["next_node"] = next_node # Set the next node
    updated_state["agent_response"] = response # Keep the agent response
    updated_state["awaiting_tenant_input_field"] = awaiting_tenant_input_field
    
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
    elif awaiting_tenant_input_field:
        return updated_state
    else:
        return AgentState(
            user_query=new_query,
            buffer_history = state.get("buffer_history"),
            summary_history=state.get("summary_history"),
            intent=None,  # Reset for new intent classification
            selected_tools=[],
            tool_outputs={},
            response=None,
            next_node="input_node"  # Loop back to process the new query
        )