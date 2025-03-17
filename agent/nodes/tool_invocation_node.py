from agent.agent_state import AgentState
from typing import Dict, Any, List
from agent.tools.vector_db_search_tool import VectorDBSearchTool
from agent.nodes.tool_selection_node import ToolName
from agent.tools.sql_tool import SQLDatabaseTool
from config.config_loader import load_config
from agent.nodes.intent_router_node import IntentCategory
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils.database_utils import load_database_schema_from_cache

import re
from langchain_core.output_parsers import BaseOutputParser

config = load_config()
db_schema_str = load_database_schema_from_cache()
llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key)

class SQLOutputParser(BaseOutputParser[str]):
    """
    A dedicated SQL output parser that validates and formats the SQL query string.
    It ensures the output starts with a valid SQL command and trims any unwanted whitespace,
    markdown code blocks, and other artifacts commonly found in LLM responses.
    """
    
    def parse(self, text: str) -> str:
        # First, remove any markdown code blocks
        # Pattern matches triple backticks with optional language identifier and any content inside
        clean_text = re.sub(r'```(?:sql)?\s*([\s\S]*?)```', r'\1', text)
        
        # Remove any potential output or response formatting
        clean_text = re.sub(r'SQL Query:|Query:|Output:', '', clean_text, flags=re.IGNORECASE)
        
        # Remove extra whitespace, newlines, and trailing semicolons
        query = clean_text.strip().rstrip(";").strip()
        
        if not query:
            raise ValueError("No SQL query output provided.")
        
        # Validate that the query begins with one of the expected SQL commands
        valid_commands = ("SELECT", "UPDATE", "INSERT", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "BEGIN", "COMMIT", "ROLLBACK", "WITH")
        # Use a regex to match the beginning of the query, ignoring case and allowing whitespace
        pattern = re.compile(rf"^\s*({'|'.join(valid_commands)})\b", re.IGNORECASE)
        
        if not pattern.match(query):
            raise ValueError(f"Output does not look like a valid SQL query: {query}")
        
        # Add back a single semicolon to the end
        query = query + ";"
        
        # Final cleanup to ensure consistent formatting
        # Replace multiple whitespaces with a single space
        query = re.sub(r'\s+', ' ', query)
        # Ensure newlines after major SQL keywords for readability
        for cmd in ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "INSERT INTO", "VALUES", "UPDATE", "SET", "DELETE FROM"]:
            query = re.sub(rf'({cmd})\s+', rf'\1\n', query, flags=re.IGNORECASE)
        
        return query

    @property
    def _type(self) -> str:
        return "sql_output_parser"

sql_output_parser = SQLOutputParser()

def generate_sql_query(user_query: str, intent: IntentCategory, conversation_history: str) -> str:
    """
    Generates dynamic SQL query using LLM based on user query and intent,
    with improved handling for vague queries.
    """
    print("--- generate_dynamic_sql_query ---")
    
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
            
            Strictly use the fields which are in Database Schema don't try to use other field names

            Conversation Context:
            {conversation_history}

            User Query: {user_query}
            User Intent: {intent}
            
            ## Handling Vague Queries:
            When the user query is vague (like "where can I find iPhone" or "show me restaurants"):
            1. First, analyze the schema to determine which tables and columns are most relevant
            2. For product or brand-related vague queries:
               - Check if the product/brand exists in stores.store_name, stores.category, or offers tables
               - Use LIKE operators with wildcards for partial matches (e.g., '%iPhone%')
               - Search across multiple relevant columns (store_name, category, brand_name, etc.)
            3. For location-based vague queries:
               - Join with the malls table to provide mall information
               - If no specific mall is mentioned, search across all malls
            4. For time/event-based vague queries:
               - Use date functions for current events if needed
            
            ## Intent Interpretation:
            - SELECT: When the user is asking for information
            - UPDATE: When the user intends to update existing data
            - INSERT: When the user wants to add new data
            - DELETE: When the user wants to remove data

            ## Query Construction Guidelines:
            - Construct queries that JOIN appropriate tables when information spans multiple tables
            - Use proper aliasing for clarity in complex queries
            - Include all relevant columns that would answer the user's query comprehensively
            - For vague product/service queries, use LIKE operator with wildcards (e.g., WHERE store_name LIKE '%iPhone%' OR category LIKE '%electronics%')
            - For list-type queries, include ORDER BY clauses for logical presentation
            - Always limit results to a reasonable number (e.g., LIMIT 10) for readability
            - When appropriate, use WITH clauses for complex queries to improve readability

            ## Column Importance Guidelines:
            - For stores: prioritize store_name, category, mall_name, location_in_mall, operating_hours
            - For offers: prioritize offer_name, description, discount_percentage, brand_name, end_date
            - For events: prioritize title, description, event_date, start_time, end_time, mall_name
            - For malls: prioritize name, location, operating_hours

            Return ONLY the SQL query as plain text. Do not include any explanation or comments.

            ## Example Vague Query Handling:
            
            User Query: "where can i find iphone"
            SQL Query: SELECT s.store_name, s.category, m.name as mall_name, s.location_in_mall, s.operating_hours 
            FROM stores s 
            JOIN malls m ON s.mall_id = m.id 
            WHERE s.store_name LIKE '%iPhone%' 
            OR s.store_name LIKE '%Apple%' 
            OR s.category LIKE '%electronics%' 
            OR EXISTS (SELECT 1 FROM offers o WHERE o.store_id = s.id AND (o.offer_name LIKE '%iPhone%' OR o.description LIKE '%iPhone%')) 
            ORDER BY m.name, s.store_name 
            LIMIT 15;
            
            User Query: "show me restaurants"
            SQL Query: SELECT s.store_name, m.name as mall_name, s.location_in_mall, s.operating_hours, s.contact_phone 
            FROM stores s 
            JOIN malls m ON s.mall_id = m.id 
            WHERE s.category LIKE '%restaurant%' 
            OR s.category LIKE '%food%' 
            OR s.category LIKE '%dining%' 
            ORDER BY m.name, s.store_name 
            LIMIT 20;
            
            User Query: "events this weekend"
            SQL Query: SELECT e.title, e.description, e.event_date, e.start_time, e.end_time, m.name as mall_name 
            FROM events e 
            JOIN malls m ON e.mall_id = m.id 
            WHERE e.event_date >= CURRENT_DATE 
            AND e.event_date <= CURRENT_DATE + INTERVAL '2 days' 
            ORDER BY e.event_date, e.start_time 
            LIMIT 15;

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
        elif tool_name == ToolName.SQL_DATABASE_QUERY:
            print(f"Invoking SQL Database Tool...")
            sql_database_tool = SQLDatabaseTool() 
            conversation_history = state.get("conversation_history", "")
            
            # Additional preprocessing for vague queries
            processed_user_query = preprocess_vague_query(user_query, conversation_history)
            
            dynamic_sql_query = generate_sql_query(processed_user_query, intent, conversation_history)

            if dynamic_sql_query:
                print("Executing Dynamic SQL Query...")
                tool_output = sql_database_tool.run(dynamic_sql_query)
                tool_outputs[ToolName.SQL_DATABASE_QUERY.value] = tool_output
            else:
                fallback_message = generate_fallback_query_message(user_query)
                tool_outputs[ToolName.SQL_DATABASE_QUERY.value] = fallback_message
                print(f"SQL Query Generation Failed. Using fallback message: {fallback_message}")
        else:
            print(f"Warning: Tool '{tool_name}' is selected but no invocation logic is implemented yet.")
            tool_outputs[tool_name_str] = f"Tool '{tool_name_str}' invocation not implemented yet."
            
    next_node = "output_node"
    
    updated_state: AgentState = state.copy()
    updated_state["tool_outputs"] = tool_outputs
    updated_state["next_node"] = next_node
    
    print("Tool Invocation Node State (Updated):", updated_state)
    return updated_state

def preprocess_vague_query(user_query: str, conversation_history: str) -> str:
    """
    Preprocesses vague queries to extract more context and specificity.
    Uses conversation history for additional context if available.
    """
    # Simple preprocessing for now - can be expanded with more sophisticated NLP
    # Convert to lowercase for easier pattern matching
    query = user_query.lower()
    
    # Add common product category mappings
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
    
    # Enrich query with mappings
    for key, enrichment in product_mappings.items():
        if key in query:
            enriched_query = f"{query} {enrichment}"
            break
    
    # Extract mall names from conversation if not in query
    if "mall" not in query and conversation_history:
        mall_pattern = re.compile(r'(?:dubai mall|mall of emirates|city centre deira|ibn battuta mall|festival city mall)', re.IGNORECASE)
        mall_matches = mall_pattern.findall(conversation_history)
        if mall_matches:
            most_recent_mall = mall_matches[-1]  # Use most recently mentioned mall
            enriched_query = f"{enriched_query} in {most_recent_mall}"
    
    print(f"Preprocessed query: '{query}' → '{enriched_query}'")
    return enriched_query

def generate_fallback_query_message(user_query: str) -> str:
    """
    Generates a helpful fallback message when SQL generation fails.
    """
    return f"I couldn't find specific information about '{user_query}'. Please try asking in a different way or provide more details about what you're looking for."