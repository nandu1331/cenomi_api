from typing import Any, Dict, List, TypedDict, Optional

class AgentState(TypedDict):
    """Represents the state of the agent."""
    user_query: str
    buffer_history: Optional[List[str]]  # Optional for backward compatibility
    summary_history: Optional[str]       # Optional for backward compatibility
    conversation_history: Optional[List[Dict[str, Any]]]  # History as list of dicts
    intent: Optional[str]
    next_node: Optional[str]
    selected_tools: Optional[List[str]]
    tool_outputs: Optional[Dict[str, str]]  # Updated to Dict for clarity
    response: Optional[str]
    awaiting_tenant_input_field: Optional[str]
    tenant_data: Optional[Dict[str, Any]]
    current_field_index: Optional[int]
    tenant_main_query: Optional[str]
    user_id: Optional[str]  # Added to track user-specific state
    language: Optional[str] # Added to track language-specific state