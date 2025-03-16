from agent.agent_state import AgentState
from typing import Dict, Any

def tool_selection_node(state: AgentState) -> AgentState:
    """Placeholder for Tool Selection Node logic."""
    print("--- Tool Selection Node ---")
    intent = state.get("intent") # Get intent from state

    print(f"Intent received by Tool Selection Node: {intent}")

    # Placeholder logic: Just route to output node for now
    next_node = "output_node" # For now, always route to output node

    updated_state: AgentState = state.copy()
    updated_state["next_node"] = next_node # Set next node in state

    print("Tool Selection Node State (Updated):", updated_state)
    return updated_state