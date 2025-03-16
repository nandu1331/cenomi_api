from typing import Any, Dict, List, TypedDict, Optional

class AgentState(TypedDict):
    """Represents the state of the agent."""
    user_query: str
    buffer_history: Optional[List[str]]
    summary_history: Optional[str]
    conversation_history: List[Dict[str, Any]]
    response: str