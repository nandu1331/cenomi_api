from typing import Dict, Any, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from enum import Enum
from utils.relevance_utils import evaluate_relevance_function

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
    intent: IntentCategory = state.get("intent") # Get intent for context

    print(f"Intent received by Tool Selection Node (Relevance-Based): {intent}")

    vector_db_tool = ToolName.VECTOR_DB_SEARCH # Instantiate tools here
    sql_db_tool = ToolName.SQL_DATABASE_QUERY


    # 1. Try VectorDB Tool First
    print("Trying VectorDB Search Tool...")
    vector_db_output = vector_db_tool.run(user_query)
    print(f"VectorDB Tool Output (initial):\n{vector_db_output}")

    # 2. Evaluate Relevance of VectorDB Output (using injected function)
    print("Evaluating relevance of VectorDB output...")
    vector_db_relevance_score = evaluate_relevance_function(user_query, vector_db_output, intent) # Call injected relevance function
    print(f"VectorDB Relevance Score: {vector_db_relevance_score}")

    VECTORDB_RELEVANCE_THRESHOLD = 0.7 # Define a threshold - tune this value

    # 3. Check Relevance Threshold
    if vector_db_relevance_score >= VECTORDB_RELEVANCE_THRESHOLD:
        print("VectorDB output is relevant enough. Selecting VectorDB Tool.")
        selected_tool_names = [ToolName.VECTOR_DB_SEARCH.value] # Select VectorDB if relevant
        tool_output = {ToolName.VECTOR_DB_SEARCH.value: vector_db_output} # Store output immediately
        next_node = "llm_call_node" # Go directly to LLM Call Node (bypass tool invocation node)

    else:
        print("VectorDB output is not relevant enough. Falling back to SQL Database Tool.")
        print("Trying SQL Database Tool...")
        sql_db_output = sql_db_tool.run(user_query) # Run SQL Tool
        print(f"SQL Database Tool Output:\n{sql_db_output}")
        selected_tool_names = [ToolName.SQL_DATABASE_QUERY.value] # Select SQL Tool
        tool_output = {ToolName.SQL_DATABASE_QUERY.value: sql_db_output} # Store SQL output
        next_node = "llm_call_node" # Go directly to LLM Call Node (bypass tool invocation node)


    updated_state: AgentState = state.copy()
    updated_state["selected_tools"] = selected_tool_names # Store selected tool names (string values)
    updated_state["tool_output"] = tool_output # Store tool output directly in state - no need for separate invocation node now
    updated_state["next_node"] = next_node # Go to LLM call node directly

    print("Tool Selection Node State (Updated - Relevance-Based):", updated_state)
    return updated_state