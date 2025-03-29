from typing import Any, Dict, List, TypedDict, Optional

class AgentState(TypedDict):
    """Represents the state of the agent."""
    user_query: str
    hybrid_context: Optional[str]
    conversation_history: List[Dict[str, Any]]
    intent: Optional[str]
    next_node: Optional[str]
    selected_tools: Optional[List[str]]
    tool_outputs: Optional[str]
    response: str
    role: str
    not_allowed: bool
    user_id: int
    mall_name: str
    mall_id: int
    session_id: str