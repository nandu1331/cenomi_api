from typing import Dict, Any, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from enum import Enum
from utils.relevance_utils import evaluate_relevance_function
from tools.sql_tool import SQLDatabaseTool
from tools.vector_db_search_tool import VectorDBSearchTool

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
    Tool Selection Node: Attempts to answer with VectorDB first, checks relevance,
    falls back to SQLDatabaseTool if VectorDB is not relevant.
    """
    print("--- Tool Selection Node (Relevance-Based) ---")
    user_query: str = state.get("user_query")
    intent: IntentCategory = state.get("intent")

    print(f"Intent received by Tool Selection Node (Relevance-Based): {intent}")

    vector_db_tool = VectorDBSearchTool()
    sql_db_tool = SQLDatabaseTool()
    
    selected_tool_names = []
    tool_output = ""

    list_keywords = ["list", "show", "what are", "what is", "give me", "find me", "display"]
    filter_keywords = ["where", "which", "that", "specific", "conditional", "only", "just"]

    query_lower = user_query.lower()
    
    is_list_query = any(keyword in query_lower for keyword in list_keywords)
    is_filter_query = any(keyword in query_lower for keyword in filter_keywords)

    if is_list_query or is_filter_query or intent in [ # Also check intent category for list hints (as fallback or reinforcement)
        IntentCategory.LIST_MALLS,
        IntentCategory.LIST_STORES_IN_MALL,
        IntentCategory.LIST_SERVICES_IN_MALL,
        IntentCategory.LIST_EVENTS_IN_MALL,
        IntentCategory.CUSTOMER_QUERY_EVENT_INFO, # Add event info queries too, as lists might be expected
        IntentCategory.CUSTOMER_QUERY_SERVICE_QUERY # Add service queries too
    ]:
        print("List or Filter keywords/intent detected. Prioritizing SQL Database Tool.")
        sql_db_output = sql_db_tool.run(user_query) # Run SQL Tool directly
        print(f"SQL Database Tool Output:\n{sql_db_output}")
        selected_tool_names = [ToolName.SQL_DATABASE_QUERY.value] # Force SQL selection
        tool_output = {ToolName.SQL_DATABASE_QUERY.value: sql_db_output}
        next_node = "llm_call_node"

    # 2. Relevance-Based Selection for Other Customer Queries (if not list/filter query)
    elif intent in [
        IntentCategory.CUSTOMER_QUERY,
        IntentCategory.CUSTOMER_QUERY_MALL_INFO,
        IntentCategory.CUSTOMER_QUERY_BRAND_INFO,
        IntentCategory.CUSTOMER_QUERY_OFFER_INFO,
        IntentCategory.CUSTOMER_QUERY_STORE_QUERY,
        IntentCategory.CUSTOMER_QUERY_SPECIFIC_STORE_QUERY,
    ]:
        print("Trying VectorDB Search Tool first (relevance-based selection)...")
        vector_db_output = vector_db_tool.run(user_query)
        print(f"VectorDB Tool Output (initial):\n{vector_db_output}")

        print("Evaluating relevance of VectorDB output...")
        vector_db_relevance_score = evaluate_relevance_function(user_query, vector_db_output, intent)
        print(f"VectorDB Relevance Score: {vector_db_relevance_score}")

        VECTORDB_RELEVANCE_THRESHOLD = 0.7 # Tune as needed

        if vector_db_relevance_score >= VECTORDB_RELEVANCE_THRESHOLD:
            print("VectorDB output is relevant enough. Selecting VectorDB Tool.")
            selected_tool_names = [ToolName.VECTOR_DB_SEARCH.value]
            tool_output = {ToolName.VECTOR_DB_SEARCH.value: vector_db_output}
            next_node = "llm_call_node"
        else:
            print("VectorDB output not relevant enough. Falling back to SQL Database Tool.")
            print("Trying SQL Database Tool...")
            sql_db_output = sql_db_tool.run(user_query)
            print(f"SQL Database Tool Output:\n{sql_db_output}")
            selected_tool_names = [ToolName.SQL_DATABASE_QUERY.value]
            tool_output = {ToolName.SQL_DATABASE_QUERY.value: sql_db_output}
            next_node = "llm_call_node"

    elif intent in [
            IntentCategory.GREETING, # New: Conversational Intents Route Directly to LLM
            IntentCategory.POLITE_CLOSING
        ]:
            print(f"Routing directly to LLM call node for conversational intent: {intent}")
            next_node = "llm_call_node"
    # 3. (No Change) Routing for Tenant Action, Out-of-Scope, and Default
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
    updated_state["selected_tools"] = selected_tool_names
    updated_state["tool_output"] = tool_output
    updated_state["next_node"] = next_node

    print("Tool Selection Node State (Updated - Enhanced SQL Priority):", updated_state)
    return updated_state