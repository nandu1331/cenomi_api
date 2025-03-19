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

def generate_sql_query(user_query: str, intent: IntentCategory, conversation_history: str) -> str:
    sql_generation_prompt = ChatPromptTemplate.from_messages([
        ("system",
            """
            You are a SQL query generator for a chatbot system designed for Cenomi Malls.
            Generate accurate and efficient SQL queries for a Neon Postgres database based on user queries.
            Database Schema: {db_schema_info}
            Conversation Context: {conversation_history}
            User Query: {user_query}
            User Intent: {intent}
            Strictly use fields from the Database Schema.
            For vague queries:
            - Use LIKE with wildcards for product/brand searches.
            - Join with malls table for location-based queries.
            - Use date functions for event-based queries.
            For intent:
            - SELECT: Information requests
            - UPDATE: Update data
            - INSERT: Add data
            - DELETE: Remove data
            Construct queries with JOINs, proper aliasing, and relevant columns.
            Limit results to 10-20 for readability.
            Return ONLY the SQL query as plain text.
            """),
        ("human", "{user_query}"),
    ])
    
    sql_query_generation_chain = sql_generation_prompt | llm | sql_output_parser
    
    try:
        return sql_query_generation_chain.invoke({
            "db_schema_info": db_schema_str, "user_query": user_query,
            "conversation_history": conversation_history, "intent": intent
        })
    except Exception:
        return ""

def tool_invocation_node(state: AgentState) -> AgentState:
    selected_tool_names = state.get("selected_tools", [])
    tool_outputs = {}
    user_query = state["user_query"]
    intent = state["intent"]
    
    conversation_history_list = state.get("conversation_history", [])
    conversation_history_str = "\n".join(
        [f"User: {turn['user']}\nAssistant: {turn['bot']}" 
         for turn in conversation_history_list if 'user' in turn and 'bot' in turn]
    ) or "No previous conversation history available."
    
    for tool_name_str in selected_tool_names:
        try:
            tool_name = ToolName(tool_name_str)
            if tool_name == ToolName.VECTOR_DB_SEARCH:
                vector_db_search_tool = VectorDBSearchTool()
                tool_output = vector_db_search_tool.run(user_query)
                tool_outputs[ToolName.VECTOR_DB_SEARCH.value] = tool_output
            elif tool_name == ToolName.SQL_DATABASE_QUERY:
                sql_database_tool = SQLDatabaseTool()
                processed_user_query = preprocess_vague_query(user_query, conversation_history_str)
                dynamic_sql_query = generate_sql_query(processed_user_query, intent, conversation_history_str)
                if dynamic_sql_query:
                    tool_output = sql_database_tool.run(dynamic_sql_query)
                    tool_outputs[ToolName.SQL_DATABASE_QUERY.value] = tool_output
                else:
                    tool_outputs[ToolName.SQL_DATABASE_QUERY.value] = "I couldn’t generate a precise query. Could you specify which item or store you’re asking about?"
        except Exception as e:
            tool_outputs[tool_name_str] = f"Sorry, I encountered an issue: {str(e)}. Please try again or provide more details!"
    
    updated_state = state.copy()
    updated_state["tool_outputs"] = tool_outputs
    updated_state["next_node"] = "llm_call_node"
    return updated_state

def preprocess_vague_query(user_query: str, conversation_history: str) -> str:
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
            enriched_query = f"{enriched_query} in {mall_matches[-1]}"
    
    return enriched_query