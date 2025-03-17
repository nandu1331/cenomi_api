from typing import Dict, Any, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory # This is our sql_database_query tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config
from utils.database_utils import load_database_schema_from_cache
from utils.extract_fields_from_query import extract_fields_from_query
from utils.get_required_fields import RequiredFieldsCalculator
from utils.primary_key_handler import PrimaryKeyHandler
from agent.tools.sql_tool import SQLDatabaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

config = load_config()
# llm = init_chat_model(model="llama3-70b-8192", model_provider="groq")
api_key = "AIzaSyCsm_mqOBKXeu72mdRAzQUqLptlWjMiJ6o"
    # llm = init_chat_model(model="llama3-70b-8192", model_provider="groq")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
output_parser = StrOutputParser()
db_schema_str = load_database_schema_from_cache()
required_fields_cal = RequiredFieldsCalculator()

def tenant_action_node(state: AgentState) -> AgentState:
    """
    Tenant Action Node: Handles tenant data manipulation (CRUD) operations.
    """
    print("--- Tenant Action Node ---")
    intent: IntentCategory = state.get("intent")
    user_query: str = state.get("user_query")
    main_query: str = state.get("tenant_main_query")

    print(f"Tenant Action Node - Intent: {intent}, User Query: {main_query}")

    updated_state: AgentState = state.copy()
    if updated_state.get("current_field_index") is None:
        updated_state["current_field_index"] = 0
        
    return handle_tenant_data_operation(updated_state, intent, user_query, main_query)

def handle_tenant_data_operation(state: AgentState, intent: IntentCategory, user_query: str, main_query: str) -> AgentState:
    """
    Generalized handler for tenant data operations on various entities.
    This version updates tenant_data from the current input if we're awaiting a field,
    then checks all required fields. Only if a field is missing does it prompt and return;
    otherwise, it proceeds with SQL generation and execution.
    """
    print("--- handle_tenant_data_operation ---")
    if state.get("awaiting_tenant_input_field") is None:
        state["tenant_data"] = {}
        
    if "tenant_data" not in state:
        state["tenant_data"] = {}
    tenant_data: Dict[str, Any] = state.get("tenant_data", {})
    entity_type = None
    operation_type = None
    required_fields: List[str] = []
    current_field_index: int = state.get("current_field_index", 0)
    
    if intent == IntentCategory.TENANT_UPDATE_OFFER:
        entity_type = "offer"
        operation_type = "update"

    elif intent == IntentCategory.TENANT_INSERT_OFFER:
        entity_type = "offer"
        operation_type = "insert"

    elif intent == IntentCategory.TENANT_DELETE_OFFER:
        entity_type = "offer"
        operation_type = "delete"
    elif intent == IntentCategory.TENANT_INSERT_STORE:
        entity_type = "store"
        operation_type = "insert"
    elif intent == IntentCategory.TENANT_UPDATE_STORE:
        entity_type = "store"
        operation_type = "update"
    else:
        print(f"handle_tenant_data_operation - No entity/operation mapping for intent: {intent}. Defaulting to LLM call node.")
        state["next_node"] = "llm_call_node"
        return state

    extracted_data = extract_fields_from_query(user_query, entity_type)
    extracted_fields = ", ".join(extracted_data.keys())
    state["tenant_data"].update(extracted_data)
    print(f"handle_tenant_data_operation - Initial Extracted Data: {state.get('tenant_data')}")
    
    if not current_field_index:
        required_fields = required_fields_cal.calculate_required_fields(intent, main_query, db_schema_str, extracted_fields)
        print(f"handle_tenant_data_operation - Calculated Required Fields: {required_fields}")
    
    pk_handler = PrimaryKeyHandler(db_schema_str, SQLDatabaseTool)
    
    state["tenant_data"] = pk_handler.handle_primary_keys(
        required_fields,
        entity_type,
        operation_type,
        state["tenant_data"]
    )
    
    print("Handled primary key: \n", tenant_data)
    
    missing_fields = [
        field for field in required_fields 
        if field not in state.get("tenant_data", {}) and not pk_handler.is_primary_key(field, entity_type)
    ]
    
    while missing_fields:
        field = missing_fields[0] # Get the first missing field
        if field not in state.get("tenant_data", {}): # Double check if field is really missing - IMPORTANT
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", f"You are assisting a tenant to {operation_type} a {entity_type}. "
                           f"We need more information. Please ask for the **{field}**."),
                ("human", f"Could you please provide the **{field}** for the {operation_type} {entity_type}?")
            ])
            prompt_chain = prompt_template | llm | output_parser
            prompt_message = prompt_chain.invoke({"operation_type": operation_type, "entity_type": entity_type, "field": field})

            print(f"handle_tenant_data_operation - Prompting for missing field '{field}': {prompt_message}")
            updated_state = state.copy()
            updated_state["agent_response"] = prompt_message
            updated_state["next_node"] = "output_node"
            updated_state["awaiting_tenant_input_field"] = field
            print(f"handle_tenant_data_operation - Returning state to prompt for '{field}'.")
            return updated_state # Return to get tenant input

        else: # Field somehow already collected (defensive check)
            print(f"handle_tenant_data_operation - Field '{field}' was missing but is now collected. Removing from missing_fields.")
            missing_fields.pop(0) # Re

    print(f"handle_tenant_data_operation - Finished field collection WHILE loop. All required fields collected in tenant_data: {tenant_data}, Proceeding to query generation...")

    try:
        print("handle_tenant_data_operation - Generating user query string for tool workflow...")
        
        enhanced_user_query_prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""
                You are an expert at generating natural language user queries for a database interaction system.
                Your task is to create a user query that accurately and informatively describes a tenant's request to perform a data operation on a mall database.

                **Database Schema:**
                {{db_schema}}

                **Tenant Operation:** {{operation_type}} {{entity_type}}

                **Tenant Data:**
                {{tenant_data}}

                **Instructions:**
                - Based on the Database Schema, Tenant Operation, and Tenant Data, generate a concise and natural user query string that a database query tool can understand.
                - The user query should clearly specify the operation type (insert, update, delete), the entity type (offer, store, etc.), and all the relevant field values from the Tenant Data.
                - The goal is to create a user-friendly query that can be used to retrieve or manipulate data in the database using a SQL query tool.
                - Example: "Insert a new offer named 'Summer Sale' with description 'Clothing sale for summer', discount percentage 30%, start date 2025-03-15, and end date 2025-03-20 in the offers table described in the schema."
                """),
            ("human", "Generate a user query string for the tenant operation.")
        ])
        
        enhanced_user_query_chain = enhanced_user_query_prompt_template | llm | output_parser
        enhanced_user_query = enhanced_user_query_chain.invoke({
            "operation_type": operation_type,
            "entity_type": entity_type,
            "tenant_data": tenant_data,
            "db_schema": db_schema_str  # Pass database schema to the prompt
        })

        print(f"handle_tenant_data_operation - Generated Enhanced User Query String: {enhanced_user_query}")

        updated_state = state.copy()
        print("Enhanced User Query: \n", enhanced_user_query)
        updated_state["user_query"] = enhanced_user_query # Set the LLM-generated user query
        updated_state["next_node"] = "tool_selection_node" # Route to tool selection
        updated_state["current_field_index"] = None
        updated_state["awaiting_tenant_input_field"] = None
        return updated_state


    except Exception as e:

        error_message = f"Error in generating user query string: {e}"

        print(error_message)

        error_response_template = ChatPromptTemplate.from_messages([

            ("system", f"Chatbot error during tenant {operation_type} operation (user query generation)."),

            ("human", "Error during user query generation for {operation_type} operation. Details: {error_message}")

        ])

        error_chain = error_response_template | llm | output_parser

        error_response = error_chain.invoke({

            "operation_type": operation_type,

            "entity_type": entity_type,

            "error_message": error_message

        })


        updated_state = state.copy()

        updated_state["agent_response"] = error_response

        updated_state["next_node"] = "output_node"
        updated_state["current_field_index"] = None # Reset field index on error as well - IMPORTANT
        updated_state["awaiting_tenant_input_field"] = None
        return updated_state