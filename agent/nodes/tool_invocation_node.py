from agent.agent_state import AgentState
from typing import Dict, Any, List
from agent.tools.vector_db_search_tool import VectorDBSearchTool
from agent.nodes.tool_selection_node import ToolName
from agent.tools.sql_tool import SQLDatabaseTool
from agent.utils.generate_sql_queries import generate_sql_query
import re

def tool_invocation_node(state: AgentState) -> AgentState:
    """
        Tool Invocation Node: Invokes the selected tool and updates the state.
    """
    #("--- Tool Invocation Node ---")
    selected_tool_names: List[str] = state.get("selected_tools", [])
    
    tool_outputs: Dict[str, str] = {}
    
    user_query = state["user_query"]
    intent = state["intent"]
    role = state.get("role", "GUEST")
    user_id = state.get("user_id")
    mall_name = state.get("mall_name")
    conversation_history = state.get("conversation_history_str", "")
    
    for tool_name_str in selected_tool_names:
        tool_name = ToolName(tool_name_str)
        
        if tool_name == ToolName.VECTOR_DB_SEARCH:
            vector_db_search_tool = VectorDBSearchTool()
            user_query = state["user_query"]
            hybrid_context = state.get("conversation_history", "")
            tool_output = vector_db_search_tool.invoke(user_query, context=hybrid_context, mall_name=mall_name)
            tool_outputs[ToolName.VECTOR_DB_SEARCH.value] = tool_output
        elif tool_name == ToolName.SQL_DATABASE_QUERY:
            sql_database_tool = SQLDatabaseTool()
            
            processed_user_query = preprocess_vague_query(user_query, conversation_history)
            
            dynamic_sql_query = generate_sql_query(processed_user_query, intent, conversation_history, mall_name)

            if dynamic_sql_query:
                tool_output = sql_database_tool.run(dynamic_sql_query)
                tool_outputs[ToolName.SQL_DATABASE_QUERY.value] = tool_output
            else:
                fallback_message = generate_fallback_query_message(user_query)
                tool_outputs[ToolName.SQL_DATABASE_QUERY.value] = fallback_message
        else:
            tool_outputs[tool_name_str] = f"Tool '{tool_name_str}' invocation not implemented yet."
            
    next_node = "output_node"
    
    updated_state: AgentState = state.copy()
    updated_state["tool_outputs"] = tool_outputs
    updated_state["next_node"] = next_node
    
    return updated_state

def preprocess_vague_query(user_query: str, conversation_history: str) -> str:
    """
    Preprocesses vague queries to extract more context and specificity.
    Uses conversation history for additional context if available.
    """
    query = user_query.lower()
    
    product_mappings = {
        "iphone": "Apple iPhone electronics mobile phones",
        "samsung": "Samsung electronics mobile phones",
        "food": "restaurants cafes dining food_court",
        "eat": "restaurants cafes dining food_court",
        "clothes": "fashion apparel clothing",
        "shoes": "footwear shoes sneakers",
        "movie": "cinema theaters entertainment",
        "watch": "cinema theaters entertainment watches jewelry",
        "play": "entertainment gaming arcade",
        "kids": "children toys entertainment family"
    }
    
    enriched_query = query
    
    for key, enrichment in product_mappings.items():
        if key in query:
            enriched_query = f"{query} {enrichment}"
            break
    
    if "mall" not in query and conversation_history:
        mall_pattern = re.compile(r'(?:dubai mall|mall of emirates|city centre deira|ibn battuta mall|festival city mall)', re.IGNORECASE)
        mall_matches = mall_pattern.findall(conversation_history)
        if mall_matches:
            most_recent_mall = mall_matches[-1]
            enriched_query = f"{enriched_query} in {most_recent_mall}"
    
    return enriched_query

def generate_fallback_query_message(user_query: str) -> str:
    """
    Generates a helpful fallback message when SQL generation fails.
    """
    return f"I couldn't find specific information about '{user_query}'. Please try asking in a different way or provide more details about what you're looking for."