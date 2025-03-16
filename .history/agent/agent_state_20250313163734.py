from typing import Any, Dict, List, TypedDict

class AgentState(TypedDict):
    """Represents the state of the agent."""
    user_query: str
    conversation_history: List[Dict[str, Any]]
    response: str