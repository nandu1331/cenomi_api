import re
from langchain_core.output_parsers import BaseOutputParser
from agent.nodes.intent_router_node import IntentCategory
from langchain.prompts import ChatPromptTemplate

from config.config_loader import load_config
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils.database_utils import load_database_schema_from_cache

config = load_config()
llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key)
db_schema_str = load_database_schema_from_cache()

class SQLOutputParser(BaseOutputParser[str]):
    """
    A dedicated SQL output parser that validates and formats the SQL query string.
    It ensures the output starts with a valid SQL command and trims any unwanted whitespace,
    markdown code blocks, and other artifacts commonly found in LLM responses.
    """
    
    def parse(self, text: str) -> str:
        clean_text = re.sub(r'```(?:sql)?\s*([\s\S]*?)```', r'\1', text)
        
        clean_text = re.sub(r'SQL Query:|Query:|Output:', '', clean_text, flags=re.IGNORECASE)
        
        query = clean_text.strip().rstrip(";").strip()
        
        if not query:
            raise ValueError("No SQL query output provided.")
        
        valid_commands = ("SELECT", "UPDATE", "INSERT", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "BEGIN", "COMMIT", "ROLLBACK", "WITH")
        pattern = re.compile(rf"^\s*({'|'.join(valid_commands)})\b", re.IGNORECASE)
        
        if not pattern.match(query):
            raise ValueError(f"Output does not look like a valid SQL query: {query}")
        
        query = query + ";"
        
        query = re.sub(r'\s+', ' ', query)
        for cmd in ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "INSERT INTO", "VALUES", "UPDATE", "SET", "DELETE FROM"]:
            query = re.sub(rf'({cmd})\s+', rf'\1\n', query, flags=re.IGNORECASE)
        
        return query

    @property
    def _type(self) -> str:
        return "sql_output_parser"

sql_output_parser = SQLOutputParser()

def generate_sql_query(user_query: str, intent: IntentCategory, conversation_history: str= "", mall_name: str=None) -> str:
    """
    Generates dynamic SQL query using LLM based on user query and intent,
    with improved handling for vague queries.
    """
    
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
            
            NOTE: Make sure the unique field values like NAMES or TITLES to be in TITLE case while generating SQL QUERY.
            
            IMPORTATNT: Only use fields in the Database Schema provided while creating the SQL query, Don't use fields specified in the 
            if they are not present in the Database Schema. Use appropriate field which matches the non existing field the query specified from the Database Schema.

            Conversation Context:
            {conversation_history}

            User Query: {user_query}
            User Intent: {intent}
            Mall Name: {mall_name}
            
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
        sql_query = sql_query_generation_chain.invoke({"db_schema_info": db_schema_str, "user_query": user_query, "conversation_history": conversation_history, "intent": intent, "mall_name": mall_name})
        return sql_query

    except Exception as e:
        error_message = f"Error generating SQL query: {e}"
        print(error_message)
        return ""