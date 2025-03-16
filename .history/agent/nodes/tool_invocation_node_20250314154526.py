from agent_state import AgentState
from typing import Dict, Any, List
from tools.vector_db_search_tool import VectorDBSearchTool
from nodes.tool_selection_node import ToolName
from tools.sql_tool import SQLDatabaseTool

def tool_invocation_node(state: AgentState) -> AgentState:
    """
        Tool Invocation Node: Invokes the selected tool and updates the state.
    """
    print("--- Tool Invocation Node ---")
    selected_tool_names: List[str] = state.get("selected_tools", [])
    
    print(f"Tools selected for invocation: {selected_tool_names}")
    
    tool_outputs: Dict[str, str] = {}
    
    for tool_name_str in selected_tool_names:
        tool_name = ToolName(tool_name_str)
        
        if tool_name == ToolName.VECTOR_DB_SEARCH:
            print(f"Invoking VectorDB Search Tool...")
            vector_db_search_tool = VectorDBSearchTool()
            user_query = state["user_query"]
            tool_output = vector_db_search_tool.invoke(user_query)
            tool_outputs[ToolName.VECTOR_DB_SEARCH.value] = tool_output
            print(f"VectorDB Search Tool Output:\n{tool_output}")
        elif tool_name == ToolName.SQL_DB_SEARCH:
            print(f"Invoking SQL Database Tool...")
            sql_database_tool = SQLDatabaseTool() # Instantiate SQL tool
            sql_query = f"SELECT title, description, event_date, start_time, end_time FROM events_view WHERE mall_name = 'Hail City Mall'" # Example SQL query (HARDCODED for now - needs to be dynamic)
            tool_output = sql_database_tool.run(sql_query) # Invoke SQL tool with query
            tool_outputs[ToolName.SQL_DB_SEARCH.value] = tool_output # Store SQL output
            print(f"SQL Database Tool Output:\n{tool_output}")
        else:
            print(f"Warning: Tool '{tool_name}' is selected but no invocation logic is implemented yet.")
            tool_outputs[tool_name_str] = f"Tool '{tool_name_str}' invocation not implemented yet."
            
    next_node = "output_node"
    
    updated_state: AgentState = state.copy()
    updated_state["tool_outputs"] = tool_outputs
    updated_state["next_node"] = next_node
    
    print("Tool Invocation Node State (Updated):", updated_state)
    return updated_state