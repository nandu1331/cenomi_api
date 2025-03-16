from agent_state import AgentState
from typing import Dict, Any, List
from tools.vector_db_search_tool import VectorDBSearchTool
from nodes.tool_selection_node import ToolName
from tools.sql_tool import SQLDatabaseTool
from config.config_loader import load_config
from nodes.intent_router_node import IntentCategory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model


config = load_config()

def generate_sql_query(user_query: str, intent: IntentCategory) -> str:
    """
    Generates dynamic SQL query using LLM based on user query and intent.
    """
    print("--- generate_dynamic_sql_query ---")
    print(f"User Query for SQL Query Generation: {user_query}")
    print(f"Intent for SQL Query Generation: {intent}")
    
    db_uri = config.database.db_uri
    db = SQLDatabase.from_uri(db_uri)
    db_schema_str = db.get_table_info()
    
    llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
    output_parser = StrOutputParser()
    
    sql_generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
                You are a SQL query generator for a chatbot system designed for Cenomi Malls.
                Your task is to generate accurate and efficient SQL queries for Neon Postgres database based on user queries.
                The database contains information about malls, stores, offers, events, and services.

                Database Schema Information:
                --- START SCHEMA INFO ---
                {db_schema_info}
                --- END SCHEMA INFO ---

                User Query: {user_query}

                Constraints and Guidelines:
                - Only generate SELECT queries. Do NOT generate INSERT, UPDATE, or DELETE queries.
                - The query should be for Neon Postgres database.
                - Only query the tables that are relevant to the user query.
                - Be precise in column names and table names based on the provided schema.
                - Return ONLY the SQL query as plain text. Do not include any explanation or comments.
                - If the user is asking to list events, ensure you select relevant columns like title, description, event_date, start_time, end_time.
                - If the user is asking about store details, select relevant store information.
                - If no relevant table or column is found for the user query, or if the query is ambiguous or cannot be translated to SQL, return an empty string.

                Example User Queries and SQL:
                User Query: "List events at Dubai Mall"
                SQL Query: SELECT title, description, event_date, start_time, end_time FROM events_view WHERE mall_name = 'Dubai Mall'

                User Query: "What are the operating hours of Mall of Emirates?"
                SQL Query: SELECT operating_hours FROM malls WHERE name = 'Mall of Emirates'

                User Query: "Find stores in Dubai Mall that are in category electronics"
                SQL Query: SELECT store_name, category, location_in_mall FROM stores_view WHERE mall_name = 'Dubai Mall' AND category = 'electronics'

                User Query: "Any offers on shoes at City Centre Deira?"
                SQL Query: SELECT offer_description, brand_name, discount_percentage, end_date FROM offers_view WHERE mall_name = 'City Centre Deira' AND category_name = 'shoes'


                Generate SQL Query:
                """),
            ("human", "{user_query}"),
        ]
    )
    
    sql_query_generation_chain = sql_generation_prompt | llm | output_parser
    
    try:
        sql_query = sql_query_generation_chain.invoke({"db_schema_info": db_schema_str, "user_query": user_query})
        print("Generated SQL Query:\n", sql_query)
        return sql_query

    except Exception as e:
        error_message = f"Error generating SQL query: {e}"
        print(error_message)
        return ""

def tool_invocation_node(state: AgentState) -> AgentState:
    """
        Tool Invocation Node: Invokes the selected tool and updates the state.
    """
    print("--- Tool Invocation Node ---")
    selected_tool_names: List[str] = state.get("selected_tools", [])
    
    print(f"Tools selected for invocation: {selected_tool_names}")
    
    tool_outputs: Dict[str, str] = {}
    
    user_query = state["user_query"]
    intent = state["intent"]
    
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
            sql_database_tool = SQLDatabaseTool() 
            dynamic_sql_query = generate_sql_query(user_query, intent) # Generate dynamic SQL

            if dynamic_sql_query: # Check if SQL query was generated successfully
                print("Executing Dynamic SQL Query...")
                tool_output = sql_database_tool.run(dynamic_sql_query) # Invoke SQL tool with dynamic query
                tool_outputs[ToolName.SQL_DB_SEARCH.value] = tool_output # Store SQL output
                print(f"SQL Database Tool Output:\n{tool_output}")
            else:
                tool_outputs[ToolName.SQL_DATABASE_QUERY.value] = "Error: Could not generate SQL query." # Error message if SQL generation failed
                print("Error: Could not generate SQL query for SQL Database Tool.")
        else:
            print(f"Warning: Tool '{tool_name}' is selected but no invocation logic is implemented yet.")
            tool_outputs[tool_name_str] = f"Tool '{tool_name_str}' invocation not implemented yet."
            
    next_node = "output_node"
    
    updated_state: AgentState = state.copy()
    updated_state["tool_outputs"] = tool_outputs
    updated_state["next_node"] = next_node
    
    print("Tool Invocation Node State (Updated):", updated_state)
    return updated_state