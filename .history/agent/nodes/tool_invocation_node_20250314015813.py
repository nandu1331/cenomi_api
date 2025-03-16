from agent.agent_state import AgentState
from typing import Dict, Any

def tool_invocation_node(state: AgentState) -> AgentState:
    """Placeholder for Tool Invocation Node logic."""
    print("--- Tool Invocation Node ---")
    selected_tools = state.get("selected_tools") # Get selected tool names from state

    print(f"Tools selected for invocation: {selected_tools}")

    # Placeholder logic: Just route to output node for now
    next_node = "output_node" # Always route to output node for now

    updated_state: AgentState = state.copy()
    updated_state["tool_output"] = "Tool Invocation Node Placeholder Output" # Placeholder tool output
    updated_state["next_node"] = next_node # Set next node

    print("Tool Invocation Node State (Updated):", updated_state)
    return updated_state