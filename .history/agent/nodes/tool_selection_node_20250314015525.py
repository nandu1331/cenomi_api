from typing import Any, Dict, List, Optional
from agent_state import AgentState
from intent_router_node import IntentCategory
from enum import Enum

class ToolName(str, Enum):
    VECTOR_DB_SEARCH = "vector_db_search"

AVAILABLE_TOOLS = {
    ToolName.VECTOR_DB_SEARCH: ToolName.VECTOR_DB_SEARCH
}

def tool_selection_node(state: AgentState) -> AgentState:
    """
    Tool Selection Node: Selects appropriate tools based on the identified intent.
    """
    intent = state["intent"]
    print(f"Intent received in Tool Selection Node: {intent}")
    
    selected_tools: List[ToolName] = []
    next_node: str = "output_node"
    
    if intent in [
        IntentCategory.CUSTOMER_QUERY,
        IntentCategory.CUSTOMER_QUERY_MALL_INFO,
        IntentCategory.CUSTOMER_QUERY_BRAND_INFO,
        IntentCategory.CUSTOMER_QUERY_OFFER_INFO,
        IntentCategory.CUSTOMER_QUERY_EVENT_INFO,
        IntentCategory.CUSTOMER_QUERY_STORE_QUERY,
        IntentCategory.CUSTOMER_QUERY_SPECIFIC_STORE_QUERY,
    ]:
        print("Customer query intent detected. Selecting VectorDB Search Tool.")
        selected_tools = [ToolName.VECTOR_DB_SEARCH] # Select VectorDB Search Tool for customer queries
        next_node = "tool_invocation_node"
    elif intent == IntentCategory.TENANT_ACTION:
        print("Tenant Action intent detected. No tool selection implemented yet.")
        next_node = "output_node" 
    elif intent == IntentCategory.OUT_OF_SCOPE:
        print("Out-of-scope intent detected. No tool selection needed.")
        next_node = "output_node" # No tool needed for out-of-scope - go to output

    else:
        print(f"No specific intent matched or tool selection logic not defined for intent: {intent}. Defaulting to Output Node.")
        next_node = "output_node"
        
    updated_state: AgentState = state.copy()
    updated_state["selected_tools"] = [tool.value for tool in selected_tools] # Store list of selected tool names (strings) in state
    updated_state["next_node"] = next_node
    
    print("Tool Selection Node State (Updated):", updated_state)
    return updated_state
