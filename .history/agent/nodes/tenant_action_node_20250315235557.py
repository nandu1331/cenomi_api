from typing import Dict, Any, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from tools.sql_tool import SQLDatabaseTool  # This is our sql_database_query tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config
from utils.database_utils import load_database_schema_from_cache

config = load_config()
llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
output_parser = StrOutputParser()
db_schema_str = load_database_schema_from_cache()

def tenant_action_node(state: AgentState) -> AgentState:
    """
    Tenant Action Node: Handles tenant data manipulation (CRUD) operations.
    """
    print("--- Tenant Action Node ---")
    intent: IntentCategory = state.get("intent")
    user_query: str = state.get("user_query")

    print(f"Tenant Action Node - Intent: {intent}, User Query: {user_query}")

    updated_state: AgentState = state.copy()
    if updated_state.get("current_field_index") is None:
        updated_state["current_field_index"] = 0
        
    return handle_tenant_data_operation(updated_state, intent)

def handle_tenant_data_operation(state: AgentState, intent: IntentCategory) -> AgentState:
    """
    Generalized handler for tenant data operations on various entities.
    This version updates tenant_data from the current input if we're awaiting a field,
    then checks all required fields. Only if a field is missing does it prompt and return;
    otherwise, it proceeds with SQL generation and execution.
    """
    print("--- handle_tenant_data_operation ---")
    # For tenant operations, start with tenant_data from state if available, else empty.
    tenant_data: Dict[str, Any] = state.get("tenant_data", {})

    # Determine the operation and required fields based on intent.
    entity_type = None
    operation_type = None
    required_fields: List[str] = []
    current_field_index: int = state.get("current_field_index", 0)
    
    if intent == IntentCategory.TENANT_UPDATE_OFFER:
        entity_type = "offer"
        operation_type = "update"
        required_fields = ["title", "discount_percentage"]

    elif intent == IntentCategory.TENANT_INSERT_OFFER:
        entity_type = "offer"
        operation_type = "insert"
        required_fields = ["store_id", "product_id","title", "description", "discount_percentage", "start_date", "end_date"]

    elif intent == IntentCategory.TENANT_DELETE_OFFER:
        entity_type = "offer"
        operation_type = "delete"
        required_fields = ["offer_name"]
    elif intent == IntentCategory.TENANT_INSERT_STORE:
        entity_type = "store"
        operation_type = "insert"
        required_fields = ["mall_name", "store_name", "tenant_name", "category", "location_in_mall", "contact_phone", "operating_hours"]
    elif intent == IntentCategory.TENANT_UPDATE_STORE:
        entity_type = "store"
        operation_type = "update"
        required_fields = ["store_name", "operating_hours"]
    else:
        print(f"handle_tenant_data_operation - No entity/operation mapping for intent: {intent}. Defaulting to LLM call node.")
        state["next_node"] = "llm_call_node"
        return state

    print(f"handle_tenant_data_operation - Entity Type: {entity_type}, Operation Type: {operation_type}, Required Fields: {required_fields}, Current Field Index: {current_field_index}")
    
    while current_field_index < len(required_fields): # Loop based on index and required_fields length
        field = required_fields[current_field_index] # Get field from required_fields using index
        print(f"handle_tenant_data_operation - Starting field collection for field: {field} (Index: {current_field_index})")
        field_value = tenant_data.get(field)

        if not field_value:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", f"You are assisting a tenant with {operation_type}ing a {entity_type}. You need to collect: {', '.join(required_fields)}. Asking for: '{field}'."),
                ("human", "Tenant wants to {operation_type} {entity_type}. Required fields: {required_fields}. Still need: {field}. Ask for '{field}'.")
            ])
            prompt_chain = prompt_template | llm | output_parser

            prompt_message = prompt_chain.invoke({
                "operation_type": operation_type,
                "entity_type": entity_type,
                "required_fields": required_fields,
                "field": field
            })

            print(f"handle_tenant_data_operation - Prompting for field '{field}': {prompt_message}")
            updated_state = state.copy()
            updated_state["response"] = prompt_message
            updated_state["next_node"] = "output_node"
            updated_state["awaiting_tenant_input_field"] = field
            print(f"handle_tenant_data_operation - Returning state after prompting for '{field}'. Next Node: {updated_state['next_node']}, Awaiting Input Field: {updated_state['awaiting_tenant_input_field']}, Current Field Index: {current_field_index}")
            return updated_state # Return to get tenant input

        else:
            print(f"handle_tenant_data_operation - Field '{field}' (Index: {current_field_index}) already collected: {field_value}. Moving to next field.")
            current_field_index += 1 # Increment index if field already collected
            state["current_field_index"] = current_field_index # Update current_field_index in state - IMPORTANT

    print(f"handle_tenant_data_operation - Finished field collection WHILE loop. All required fields collected in tenant_data: {tenant_data}, Final Field Index: {current_field_index}")

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
        user_query_string = enhanced_user_query_chain.invoke({
            "operation_type": operation_type,
            "entity_type": entity_type,
            "tenant_data": tenant_data,
            "db_schema": db_schema_str  # Pass database schema to the prompt
        })

        print(f"handle_tenant_data_operation - Generated Enhanced User Query String: {user_query_string}")

        updated_state = state.copy()
        updated_state["user_query"] = user_query_string # Set the LLM-generated user query
        updated_state["next_node"] = "tool_selection_node" # Route to tool selection
        updated_state["current_field_index"] = None
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
        return updated_state