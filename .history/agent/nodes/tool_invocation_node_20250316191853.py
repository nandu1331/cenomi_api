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
from utils.database_utils import get_database_schema_string

config = load_config()
db_schema_str = get_database_schema_string(use_cache=True)

import re
from langchain_core.output_parsers import BaseOutputParser

class SQLOutputParser(BaseOutputParser[str]):
    """
    A dedicated SQL output parser that validates and formats the SQL query string.
    It ensures the output starts with a valid SQL command and trims any unwanted whitespace.
    """
    
    def parse(self, text: str) -> str:
        # Remove extra whitespace and trailing semicolons.
        query = text.strip().rstrip(";").strip() + ";"
        if not query:
            raise ValueError("No SQL query output provided.")
        
        # Validate that the query begins with one of the expected SQL commands.
        valid_commands = ("SELECT", "UPDATE", "INSERT", "DELETE")
        # Use a regex to match the beginning of the query.
        pattern = re.compile(rf"^({'|'.join(valid_commands)})", re.IGNORECASE)
        if not pattern.match(query):
            raise ValueError(f"Output does not look like a valid SQL query: {query}")
        
        return query

    @property
    def _type(self) -> str:
        return "sql_output_parser"

sql_output_parser = SQLOutputParser()

def generate_sql_query(user_query: str, intent: IntentCategory, conversation_history: str) -> str:
    """
    Generates dynamic SQL query using LLM based on user query and intent.
    """
    print("--- generate_dynamic_sql_query ---")
    print(f"User Query for SQL Query Generation: {user_query}")
    print(f"Intent for SQL Query Generation: {intent}")
    
    llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
    output_parser = StrOutputParser()
    
    
    sql_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """
            You are a SQL query generator for a chatbot system designed for Cenomi Malls.
            Your task is to generate accurate and efficient SQL queries for a Neon Postgres database based on user queries.
            The database contains information about malls, stores, offers, events, and services.

            Database Schema Information:
            --- START SCHEMA INFO ---
            {db_schema_info}
            --- END SCHEMA INFO ---

            Conversation Context:
            {conversation_history}

            User Query: {user_query}
            User Intent: {intent}
            
            Generate the queries based on the intent.
            If the intent is about UPDATE the user intends to update the row
            If the intent is about INSERT the user intends to insert a new row

            Constraints and Guidelines:
            - The query should be for a Neon Postgres database.
            - Only query the tables that are relevant to the user query.
            - Be precise in column names and table names based on the provided schema.
            - Return ONLY the SQL query as plain text. Do not include any explanation or comments.
            - For SELECT queries:
                - If the user is asking to list events, ensure you select relevant columns like title, description, event_date, start_time, end_time.
                - If the user is asking about store details, select relevant store information.
            - For CRUD operations (INSERT, UPDATE, DELETE):
                - Follow the proper SQL syntax for the respective operation.
                - Ensure that UPDATE queries include a WHERE clause to target specific records.
                - For INSERT queries, list the column names and their corresponding values.
                - For DELETE queries, include a WHERE clause to avoid deleting unintended records.
            - If no relevant table or column is found for the user query, or if the query is ambiguous or cannot be translated to SQL, return an empty string.

            Example SELECT Queries and SQL:
            User Query: "List events at Dubai Mall"
            SQL Query: SELECT title, description, event_date, start_time, end_time FROM events_view WHERE mall_name = 'Dubai Mall';

            User Query: "What are the operating hours of Mall of Emirates?"
            SQL Query: SELECT operating_hours FROM malls WHERE name = 'Mall of Emirates';

            User Query: "Find stores in Dubai Mall that are in category electronics"
            SQL Query: SELECT store_name, category, location_in_mall FROM stores_view WHERE mall_name = 'Dubai Mall' AND category = 'electronics';

            User Query: "Any offers on shoes at City Centre Deira?"
            SQL Query: SELECT offer_description, brand_name, discount_percentage, end_date FROM offers_view WHERE mall_name = 'City Centre Deira' AND category_name = 'shoes';

            Additional CRUD Examples:
            User Query: "Update the discount percentage for the Pizza Party offer to 50%"
            SQL Query: UPDATE offers SET discount_percentage = 50 WHERE offer_name = 'Pizza Party';

            User Query: "Insert a new store with name 'Cool Gadgets' in Dubai Mall, category 'electronics', contact phone '1234567890', and operating hours '10:00 AM - 10:00 PM'"
            SQL Query: INSERT INTO stores (mall_name, store_name, category, contact_phone, operating_hours) VALUES ('Dubai Mall', 'Cool Gadgets', 'electronics', '1234567890', '10:00 AM - 10:00 PM');

            User Query: "Delete the 'Summer Sale' offer"
            SQL Query: DELETE FROM offers WHERE offer_name = 'Summer Sale';

            Generate SQL Query:
            """
        ),
        ("human", "{user_query}"),
    ]
)
    
    sql_query_generation_chain = sql_generation_prompt | llm | sql_output_parser
    
    try:
        sql_query = sql_query_generation_chain.invoke({"db_schema_info": db_schema_str, "user_query": user_query, "conversation_history": conversation_history, "intent": intent})
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
            hybrid_context = state.get("conversation_history", "")
            tool_output = vector_db_search_tool.invoke(user_query, context=hybrid_context)
            tool_outputs[ToolName.VECTOR_DB_SEARCH.value] = tool_output
            print(f"VectorDB Search Tool Output:\n{tool_output}")
        elif tool_name == ToolName.SQL_DATABASE_QUERY:
            print(f"Invoking SQL Database Tool...")
            sql_database_tool = SQLDatabaseTool() 
            conversation_history = state.get("conversation_history", "")
            dynamic_sql_query = generate_sql_query(user_query, intent, conversation_history) # Generate dynamic SQL

            if dynamic_sql_query: # Check if SQL query was generated successfully
                print("Executing Dynamic SQL Query...")
                tool_output = sql_database_tool.run(dynamic_sql_query) # Invoke SQL tool with dynamic query
                tool_outputs[ToolName.SQL_DATABASE_QUERY.value] = tool_output # Store SQL output
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