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
db_schema = load_database_schema_from_cache()

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
        required_fields = ["offer_name", "discount_percentage"]

    elif intent == IntentCategory.TENANT_INSERT_OFFER:
        entity_type = "offer"
        operation_type = "insert"
        required_fields = ["offer_name", "description", "discount_percentage", "start_date", "end_date"]

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
            updated_state["agent_response"] = prompt_message
            updated_state["next_node"] = "output_node"
            updated_state["awaiting_tenant_input_field"] = field
            print(f"handle_tenant_data_operation - Returning state after prompting for '{field}'. Next Node: {updated_state['next_node']}, Awaiting Input Field: {updated_state['awaiting_tenant_input_field']}, Current Field Index: {current_field_index}")
            return updated_state # Return to get tenant input

        else:
            print(f"handle_tenant_data_operation - Field '{field}' (Index: {current_field_index}) already collected: {field_value}. Moving to next field.")
            current_field_index += 1 # Increment index if field already collected
            state["current_field_index"] = current_field_index # Update current_field_index in state - IMPORTANT

    print(f"handle_tenant_data_operation - Finished field collection WHILE loop. All required fields collected in tenant_data: {tenant_data}, Final Field Index: {current_field_index}")


    # --- 3. Dynamically Generate and Execute SQL Query using SQL_DATABASE_QUERY Tool --- (Same as before)
    try:
        print("handle_tenant_data_operation - Dynamically generating SQL query using LLM...")
        print("Tenant Data:\n", tenant_data)
        sql_generation_prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a SQL query generator for a mall management system. "
        "Generate a valid SQL UPDATE query for updating an offer in a PostgreSQL database. "
        "Use the following tenant data to construct the query: "
        "offer_name: {offer_name}, discount_percentage: {discount_percentage}. "
        "Return only the SQL query without any additional text."
    )),
    ("human", "Generate SQL query.")
])
        sql_generation_chain = sql_generation_prompt_template | llm | output_parser

        sql_query = sql_generation_chain.invoke({
            "offer_name": tenant_data["offer_name"],
            "discount_percentage": tenant_data["discount_percentage"]
        })

        print(f"handle_tenant_data_operation - Generated SQL Query from LLM: {sql_query}")

        # --- Execute SQL Query using SQL_DATABASE_QUERY tool ---
        print("handle_tenant_data_operation - Executing SQL query using SQL_DATABASE_QUERY tool...")
        updated_state = state.copy()
        updated_state["tool_code"] = "SQL_DATABASE_QUERY"
        updated_state["tool_input"] = sql_query
        updated_state["selected_tools"] = ["SQL_DATABASE_QUERY"] # Force tool selection

        print(f"handle_tenant_data_operation - State before routing to tool selection: {updated_state}")
        updated_state["next_node"] = "tool_selection_node" # Route to tool selection
        updated_state["current_field_index"] = None # Reset field index after operation is complete - IMPORTANT
        return updated_state


    except Exception as e:
        error_message = f"Error in dynamic SQL generation or execution: {e}"
        print(error_message)
        error_response_template = ChatPromptTemplate.from_messages([
            ("system", f"Chatbot error during tenant {operation_type} operation."),
            ("human", "Error during {operation_type} operation. Details: {error_message}")
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