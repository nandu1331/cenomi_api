from typing import Dict, Any, List, Callable
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from tools.sql_tool import SQLDatabaseTool # Import SQL Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config

config = load_config()

def tenant_action_node(state: AgentState) -> AgentState:
    """
    Tenant Action Node: Handles tenant data manipulation (CRUD) operations using a generalized handler.
    """
    print("--- Tenant Action Node ---")
    intent: IntentCategory = state.get("intent")
    user_query: str = state.get("user_query")

    print(f"Tenant Action Node - Intent: {intent}, User Query: {user_query}")

    updated_state: AgentState = state.copy()

    return handle_tenant_data_operation(updated_state, intent) # Call generalized handler



def handle_tenant_data_operation(state: AgentState, intent: IntentCategory) -> AgentState:
    """
    Generalized handler for tenant data operations (update, insert, delete) on various entities.
    """
    print("--- handle_tenant_data_operation ---")
    tenant_data: Dict[str, Any] = state.get("tenant_data", {}) # Get generalized tenant_data
    entity_type = None # Determine entity type based on intent
    operation_type = None # Determine operation type (update, insert, delete)

    sql_tool = SQLDatabaseTool() # Instantiate SQL tool
    llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
    output_parser = StrOutputParser()


    # --- 1. Determine Entity and Operation Type based on Intent ---
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

    elif intent == IntentCategory.TENANT_UPDATE_STORE:
        entity_type = "store"
        operation_type = "update"
        required_fields = ["store_name", "operating_hours"]
    else:
        print(f"handle_tenant_data_operation - No entity/operation mapping for intent: {intent}. Defaulting to LLM call node.")
        state["next_node"] = "llm_call_node"
        return state


    # --- 2. Collect Required Fields (Multi-turn Conversation) ---
    for field in required_fields:
        field_value = tenant_data.get(field) # Check if field already collected
        if not field_value:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", f"You are assisting a mall tenant with {operation_type}ing a {entity_type}. You need to collect the following information to proceed: {', '.join(required_fields)}. Currently, I need the '{field}'. Please ask a clear and concise question to get the value for '{field}' from the tenant."),
                ("human", "Tenant wants to {operation_type} {entity_type}. Required fields are: {required_fields}. I still need: {field}. Ask for '{field}'.")
            ])
            prompt_chain = prompt_template | llm | output_parser

            prompt_message = prompt_chain.invoke({
                "operation_type": operation_type,
                "entity_type": entity_type,
                "required_fields": required_fields,
                "field": field
            })

            print(f"handle_tenant_data_operation - Prompting for {field}: {prompt_message}")

            updated_state = state.copy()
            updated_state["agent_response"] = prompt_message
            updated_state["next_node"] = "llm_call_node"
            updated_state["awaiting_tenant_input_field"] = field # Set awaiting input field
            return updated_state # Return to get tenant input


    # --- 3. Construct and Execute SQL Query ---
    try:
        print("handle_tenant_data_operation - Constructing and executing SQL query...")
        sql_query = ""
        if operation_type == "update":
            set_clauses = ", ".join([f"{f} = '{tenant_data[f]}'" for f in required_fields if f != 'offer_name']) # Example - adjust based on fields and where clause
            sql_query = f"""
            UPDATE offers
            SET {set_clauses}
            WHERE offer_name = '{tenant_data['offer_name']}';
            """ # Construct UPDATE query - adapt to your DB schema and where clause

        elif operation_type == "insert":
            columns = ", ".join(required_fields)
            values = ", ".join([f"'{tenant_data[f]}'" for f in required_fields])
            sql_query = f"""
            INSERT INTO offers ({columns})
            VALUES ({values});
            """ # Construct INSERT query - adapt to your DB schema

        elif operation_type == "delete":
            sql_query = f"""
            DELETE FROM offers
            WHERE offer_name = '{tenant_data['offer_name']}';
            """ # Construct DELETE query - adapt to your DB schema


        print(f"handle_tenant_data_operation - Executing SQL Query: {sql_query}")
        sql_output = sql_tool.run(sql_query) # Execute SQL query
        print(f"handle_tenant_data_operation - SQL Tool Output (Operation Result): {sql_output}")

        confirmation_message_template = ChatPromptTemplate.from_messages([
            ("system", f"You are a chatbot confirming tenant's {operation_type} operation on {entity_type} data. Provide a concise confirmation message."),
            ("human", "Confirm {operation_type} operation on {entity_type}.")
        ])
        confirmation_chain = confirmation_message_template | llm | output_parser
        confirmation_message = confirmation_chain.invoke({
            "operation_type": operation_type,
            "entity_type": entity_type
        })


        updated_state = state.copy()
        updated_state["agent_response"] = confirmation_message
        updated_state["next_node"] = "llm_call_node"
        updated_state["tenant_data"] = {} # Clear tenant_data after operation
        return updated_state

    except Exception as e:
        error_message = f"Error performing tenant data operation in database: {e}"
        print(error_message)
        error_response_template = ChatPromptTemplate.from_messages([
            ("system", f"You are a chatbot handling errors during tenant's {operation_type} operation on {entity_type} data. Inform the tenant about the error concisely and ask them to try again or contact support."),
            ("human", "Error during {operation_type} operation on {entity_type}. Error details: {error_message}")
        ])
        error_chain = error_response_template | llm | output_parser
        error_response = error_chain.invoke({
            "operation_type": operation_type,
            "entity_type": entity_type,
            "error_message": error_message
        })

        updated_state = state.copy()
        updated_state["agent_response"] = error_response
        updated_state["next_node"] = "llm_call_node"
        return updated_state