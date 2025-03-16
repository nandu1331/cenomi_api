from typing import Dict, Any, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from tools.sql_tool import SQLDatabaseTool

def tenant_action_node(state: AgentState) -> AgentState:
    """
    Tenant Action Node: Handles tenant data manipulation (CRUD) operations.
    Orchestrates multi-turn conversation to gather required fields and performs DB updates.
    (Currently focused on "update offer" scenario as example).
    """
    print("--- Tenant Action Node ---")
    user_query: str = state.get("user_query")
    intent: IntentCategory = state.get("intent")
    conversation_history: List[Dict[str, str]] = state.get("conversation_history", [])
    
    print(f"Tenant Action Node - Intent: {intent}, User Query: {user_query}")
    updated_state: AgentState = state.copy()
    
    if intent == IntentCategory.TENANT_INSERT_OFFER:
        return handle_insert_offer(updated_state)
    elif intent == IntentCategory.TENANT_UPDATE_OFFER:
        return handle_update_offer(updated_state)
    elif intent == IntentCategory.TENANT_DELETE_OFFER:
        return handle_delete_offer(updated_state)
    elif intent == IntentCategory.TENANT_INSERT_STORE:
        return handle_insert_store(updated_state)
    elif intent == IntentCategory.TENANT_UPDATE_STORE:
        return handle_update_store(updated_state)
    elif intent == IntentCategory.TENANT_DELETE_STORE:
        return handle_delete_store(updated_state)
    elif intent == IntentCategory.TENANT_INSERT_EVENT:
        return handle_insert_offer(updated_state)
    elif intent == IntentCategory.TENANT_UPDATE_EVENT:
        return handle_update_offer(updated_state)
    elif intent == IntentCategory.TENANT_DELETE_EVENT:
        return handle_delete_offer(updated_state)
    