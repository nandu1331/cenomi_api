from typing import Dict, Any, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from langchain_core.prompts import ChatPromptTemplate
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
llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key)
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
    
    # Initialize tenant_data if not present
    if "tenant_data" not in state:
        state["tenant_data"] = {}
    
    # Process field input if we were awaiting a specific field
    if state.get("awaiting_tenant_input_field") is not None:
        field_response = user_query
        field_name = state.get("awaiting_tenant_input_field")
        
        # Special handling for field selection mode
        if state.get("field_selection_mode", False):
            # User has selected fields they want to update
            selected_fields = extract_fields_for_update(field_response, db_schema_str)
            if selected_fields:
                state["required_fields"] = selected_fields
                state["field_selection_mode"] = False
                state["awaiting_tenant_input_field"] = selected_fields[0]
                
                # Prepare prompt for the first field
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", f"You are assisting a tenant to update a {state['entity_type']}. "
                               f"We need more information. Please ask for the *{selected_fields[0]}*."),
                    ("human", f"Could you please provide the *{selected_fields[0]}* for the update?")
                ])
                prompt_chain = prompt_template | llm | output_parser
                prompt_message = prompt_chain.invoke({
                    "entity_type": state['entity_type'],
                    "field": selected_fields[0]
                })
                
                updated_state = state.copy()
                updated_state["response"] = prompt_message
                updated_state["next_node"] = "output_node"
                return updated_state
            else:
                # Handle case where field selection is invalid
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are assisting a tenant with data operations."),
                    ("human", "I couldn't understand which fields you want to update. Please clearly specify the fields from the available options.")
                ])
                prompt_chain = prompt_template | llm | output_parser
                prompt_message = prompt_chain.invoke({})
                
                updated_state = state.copy()
                updated_state["response"] = prompt_message
                updated_state["next_node"] = "output_node"
                return updated_state
        else:
            # Normal field value capture
            state["tenant_data"][field_name] = field_response
            state["awaiting_tenant_input_field"] = None
            print(f"handle_tenant_data_operation - Captured field '{field_name}': {field_response}")
    
    tenant_data: Dict[str, Any] = state.get("tenant_data", {})
    entity_type = None
    operation_type = None
    required_fields: List[str] = []
    current_field_index: int = state.get("current_field_index", 0)
    
    # Determine entity type and operation type based on intent
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
        state["next_node"] = "llm_call_node"
        return state
    
    # Save entity_type and operation_type in state for later use
    state["entity_type"] = entity_type
    state["operation_type"] = operation_type
    
    # Extract fields from the user query
    extracted_data = extract_fields_from_query(user_query, entity_type, db_schema_str)
    state["tenant_data"].update(extracted_data)
    extracted_fields = ", ".join(extracted_data.keys())
    
    # NEW: Detect if user already specified update fields in their query
    update_fields = []
    if operation_type == "update":
        update_fields = extract_fields_for_update(user_query, db_schema_str)
        print(f"handle_tenant_data_operation - User specified update fields: {update_fields}")
    
    # Check if we're in update mode with only an identifier (and no update fields specified)
    if operation_type == "update" and len(state["tenant_data"]) == 1 and not update_fields and not state.get("field_selection_mode", False):
        # User has only provided an identifier, need to offer field selection
        unique_identifier = list(state["tenant_data"].keys())[0]
        unique_value = state["tenant_data"][unique_identifier]
        
        # Get available fields for this entity type
        available_fields = extract_entity_fields(db_schema_str, entity_type)
        
        # Remove the unique identifier from available fields
        if unique_identifier in available_fields:
            available_fields.remove(unique_identifier)
            
        # Format the fields for presentation
        formatted_fields = ", ".join([f"'{field}'" for field in available_fields])
        
        # Create a prompt to ask user which fields they want to update
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are assisting a tenant to update a {entity_type}. "
                      f"The tenant wants to update the {entity_type} with {unique_identifier} = '{unique_value}'. "
                      f"We don't know which fields they want to update so we need to ask for the fields they want to update."),
            ("human", f"You want to update the {entity_type} with {unique_identifier} = '{unique_value}'. "
                     f"Please specify which fields you'd like to update from the following options: {formatted_fields}. "
                     f"You can list multiple fields separated by commas.")
        ])
        prompt_chain = prompt_template | llm | output_parser
        prompt_message = prompt_chain.invoke({
            "entity_type": entity_type,
            "unique_identifier": unique_identifier,
            "unique_value": unique_value,
            "formatted_fields": formatted_fields
        })
        print("Prompt Message: ", prompt_message)
        
        updated_state = state.copy()
        updated_state["response"] = prompt_message
        updated_state["next_node"] = "output_node"
        updated_state["field_selection_mode"] = True
        updated_state["awaiting_tenant_input_field"] = "field_selection"
        updated_state["available_fields"] = available_fields
        print(f"handle_tenant_data_operation - Prompting for field selection with available fields: {available_fields}")
        return updated_state
    
    # Use the user-specified update fields if available
    if operation_type == "update" and update_fields:
        print(f"handle_tenant_data_operation - Using user-specified update fields: {update_fields}")
        required_fields = update_fields
        state["required_fields"] = required_fields
    # Continue with normal workflow if not in field selection mode and no update fields specified
    elif not current_field_index and not state.get("required_fields"):
        req_fields = required_fields_cal.calculate_required_fields(intent, main_query, db_schema_str, extracted_fields)
        required_fields = [field for field in req_fields if field not in required_fields]
        state["required_fields"] = required_fields
        print(f"handle_tenant_data_operation - Calculated Required Fields: {required_fields}")
    else:
        required_fields = state.get("required_fields", [])
    
    # Check for missing fields
    missing_fields = [
        field for field in required_fields 
        if field not in state.get("tenant_data", {})
    ]
    
    if missing_fields:
        field = missing_fields[0]  # Get the first missing field
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are assisting a tenant to {operation_type} a {entity_type}. "
                       f"We need more information. Please ask for the *{field}*."),
            ("human", f"Could you please provide the *{field}* for the {operation_type} {entity_type}?")
        ])
        prompt_chain = prompt_template | llm | output_parser
        prompt_message = prompt_chain.invoke({
            "operation_type": operation_type,
            "entity_type": entity_type,
            "field": field
        })

        print(f"handle_tenant_data_operation - Prompting for missing field '{field}': {prompt_message}")
        updated_state = state.copy()
        updated_state["response"] = prompt_message
        updated_state["next_node"] = "output_node"
        updated_state["awaiting_tenant_input_field"] = field
        print(f"handle_tenant_data_operation - Returning state to prompt for '{field}'.")
        return updated_state

    print(f"handle_tenant_data_operation - All required fields collected in tenant_data: {tenant_data}, Proceeding to query generation...")

    try:
        
        enhanced_user_query_prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""
                You are an expert at generating natural language user queries for a database interaction system.
                Your task is to create a user query that accurately and informatively describes a tenant's request to perform a data operation on a mall database.

                *Database Schema:*
                {{db_schema}}

                *Tenant Operation:* {{operation_type}} {{entity_type}}

                *Tenant Data:*
                {{tenant_data}}

                *Instructions:*
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

        updated_state = state.copy()
        updated_state["user_query"] = enhanced_user_query # Set the LLM-generated user query
        updated_state["next_node"] = "tool_selection_node" # Route to tool selection
        updated_state["current_field_index"] = None
        updated_state["awaiting_tenant_input_field"] = None
        updated_state["field_selection_mode"] = False
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
        updated_state["response"] = error_response
        updated_state["next_node"] = "output_node"
        updated_state["current_field_index"] = None
        updated_state["awaiting_tenant_input_field"] = None
        updated_state["field_selection_mode"] = False
        return updated_state

def extract_entity_fields(db_schema_str: str, entity_type: str) -> List[str]:
    """
    Extract available fields for a given entity type from the database schema.
    Using LLM to parse the schema and extract the fields.
    """
    # Map entity types to likely table names
    table_mappings = {
        "offer": ["offers", "offer"],
        "store": ["stores", "store"],
        # Add more mappings as needed
    }
    
    possible_tables = table_mappings.get(entity_type, [entity_type])
    
    # Use LLM to extract fields from the schema
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
            You are a database expert. Given a database schema, extract all the column names for a specific table.
            
            Database Schema:
            {db_schema_str}
            
            Task:
            Extract all column names for the table that represents {entity_type}. The table name might be one of these: {", ".join(possible_tables)}.
            
            Return only a comma-separated list of column names without any additional text or explanations.
            Don't include any SQL keywords or types, just the column names.
            """),
        ("human", f"Extract the column names for the {entity_type} table.")
    ])
    
    prompt_chain = prompt_template | llm | output_parser
    field_list_str = prompt_chain.invoke({})
    
    # Clean up the field list and convert to a list
    field_list = [field.strip() for field in field_list_str.split(',')]
    
    print(f"extract_entity_fields - Extracted fields for {entity_type}: {field_list}")
    return field_list

def extract_fields_for_update(user_input: str, db_schema: str) -> List[str]:
    """
    Extract field names for update from user input.
    
    Args:
        user_input (str): User's response containing field names they want to update
        db_schema (str): Database schema string
        
    Returns:
        List[str]: List of field names to update
    """
    # Use LLM to extract field names from natural language input
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
            You are a text processing expert. Extract the field names that the user wants to update from their input.
            
            Database Schema:
            {db_schema}
            
            User Input:
            '{user_input}'
            
            Instructions:
            - Identify the specific fields the user wants to update.
            - For example, if the input is "I want to update the description and the discount_percentage", 
              you should extract "description, discount_percentage".
            - If the input is "Update the Summer Sale offer", you should return an empty list as no specific fields are mentioned.
            - If the input is "Update the description of Summer Sale offer", you should return "description".
            - Return only a comma-separated list of the field names, without any additional text.
            - If no specific fields are mentioned for update, return an empty string.
        """),
        ("human", "Extract the field names the user wants to update.")
    ])
    
    prompt_chain = prompt_template | llm | output_parser
    field_list_str = prompt_chain.invoke({})
    
    # Clean up the field list and convert to a list
    fields = [field.strip() for field in field_list_str.split(',') if field.strip()]
    
    print(f"extract_fields_for_update - Extracted fields from input '{user_input}': {fields}")
    return fields