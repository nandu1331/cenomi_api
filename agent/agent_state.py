from typing import Any, Dict, List, TypedDict, Optional

class AgentState(TypedDict):
    """Represents the state of the agent."""
    user_query: str
    buffer_history: Optional[List[str]]
    summary_history: Optional[str]
    conversation_history: List[Dict[str, Any]]
    intent: Optional[str]
    next_node: Optional[str]
    selected_tools: Optional[List[str]]
    tool_outputs: Optional[str]
    response: str
    awaiting_tenant_input_field: Optional[str]
    tenant_data: Dict[str, Any]
    current_field_index: Optional[int]
    tenant_main_query: str
    field_selection_mode: bool
    available_fields: str
    entity_type: str
    operation_type: str
    required_fields: List
    