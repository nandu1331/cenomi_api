from typing import Dict, Any, List
from agent.agent_state import AgentState
from agent.nodes.intent_router_node import IntentCategory
from enum import Enum

# Define Tool Names
class ToolName(str, Enum):
    VECTOR_DB_SEARCH = "vector_db_search_tool"
    SQL_DATABASE_QUERY = "sql_database_query_tool"


# Define Available Tools
AVAILABLE_TOOLS = {
    ToolName.VECTOR_DB_SEARCH: ToolName.VECTOR_DB_SEARCH,
    ToolName.SQL_DATABASE_QUERY: ToolName.SQL_DATABASE_QUERY
}

def tool_selection_node(state: AgentState) -> AgentState:
    """
    Tool Selection Node: Selects appropriate tools based on the identified intent.
    Prioritizes SQLDatabaseTool for 'list_' intents.
    """
    print("--- Tool Selection Node ---")
    intent: IntentCategory = state.get("intent")

    print(f"Intent received by Tool Selection Node: {intent}")

    selected_tools: List[ToolName] = []
    next_node: str = "output_node" 

    if intent in [
        IntentCategory.CUSTOMER_QUERY,
        IntentCategory.CUSTOMER_QUERY_MALL_INFO,
        IntentCategory.CUSTOMER_QUERY_BRAND_INFO,
        IntentCategory.CUSTOMER_QUERY_OFFER_INFO,
        IntentCategory.CUSTOMER_QUERY_STORE_QUERY,
        IntentCategory.CUSTOMER_QUERY_SPECIFIC_STORE_QUERY,
        IntentCategory.CUSTOMER_QUERY_SERVICE_QUERY,
        IntentCategory.CUSTOMER_QUERY_EVENT_INFO
    ]:
        print("Customer query intent detected. Selecting VectorDB Search Tool (default).")
        selected_tools = [ToolName.VECTOR_DB_SEARCH]
        next_node = "tool_invocation_node"
    elif intent in [
        IntentCategory.LIST_MALLS,
        IntentCategory.LIST_STORES_IN_MALL,
        IntentCategory.LIST_SERVICES_IN_MALL,
        IntentCategory.LIST_EVENTS_IN_MALL,
    ]:
        print("List intent detected. Selecting SQL Database Tool (REQUIRED for list intents).")
        selected_tools = [ToolName.SQL_DATABASE_QUERY]
        next_node = "tool_invocation_node"
    elif intent == IntentCategory.CUSTOMER_QUERY_SERVICE_QUERY:
        print("Customer service query intent detected. Selecting VectorDB Search Tool (can use SQL too).")
        selected_tools = [ToolName.VECTOR_DB_SEARCH] 
        next_node = "tool_invocation_node"
    elif intent == IntentCategory.CUSTOMER_QUERY_EVENT_INFO:
        print("Customer event info query detected. Selecting VectorDB Search Tool (can use SQL too).")
        selected_tools = [ToolName.VECTOR_DB_SEARCH]
        next_node = "tool_invocation_node"
    elif intent == IntentCategory.TENANT_ACTION:
        print("Tenant Action intent detected. No tool selection implemented yet.")
        next_node = "llm_call_node"
    elif intent == IntentCategory.OUT_OF_SCOPE:
        print("Out-of-scope intent detected. No tool selection needed.")
        next_node = "llm_call_node"
    else:
        print(f"No specific intent matched or tool selection logic not defined for intent: {intent}. Defaulting to LLM Call Node.")
        next_node = "llm_call_node"


    updated_state: AgentState = state.copy()
    updated_state["selected_tools"] = [tool.value for tool in selected_tools] 
    updated_state["next_node"] = next_node 

    print("Tool Selection Node State (Updated):", updated_state)
    return updated_state