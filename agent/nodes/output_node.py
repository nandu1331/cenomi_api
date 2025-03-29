from typing import Dict, Any
from agent.agent_state import AgentState
from langgraph.graph import END

def output_node(state: Dict[str, Any]) -> AgentState:
    """Output node: Prepares the final response."""
    
    response = state.get("response", "No response generated.")
    print(f"Agent Response:\n {response}")

    return AgentState(
            user_query="",
            buffer_history = state.get("buffer_history"),
            summary_history=state.get("summary_history"),
            intent=None,  # Reset for new intent classification
            selected_tools=[],
            tool_outputs={},
            response=response,
            next_node=END
        )