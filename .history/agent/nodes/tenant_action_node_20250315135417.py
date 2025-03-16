from typing import Dict, Any, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from tools.sql_tool import SQLDatabaseTool  # This is our sql_database_query tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config

config = load_config()
llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
output_parser = StrOutputParser()

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
    
    if intent == IntentCategory.TENANT_INSERT_STORE:
        entity_type = "store"
        operation_type = "insert"
        required_fields = [
            "mall_name", "store_name", "tenant_name",
            "category", "location_in_mall", "contact_phone", "operating_hours"
        ]
    elif intent == IntentCategory.TENANT_UPDATE_STORE:
        entity_type = "store"
        operation_type = "update"
        required_fields = ["store_name", "operating_hours"]
    # (Add other tenant intents as needed)
    else:
        print(f"handle_tenant_data_operation - No mapping for intent: {intent}. Defaulting to LLM call node.")
        state["next_node"] = "llm_call_node"
        return state

    # --- Step 1: Update tenant_data if we're waiting for a specific field ---
    if "awaiting_tenant_input_field" in state:
        waiting_field = state.pop("awaiting_tenant_input_field")
        tenant_data[waiting_field] = state["user_query"].strip()
        print(f"Updated tenant_data[{waiting_field}]: {tenant_data[waiting_field]}")
        state["tenant_data"] = tenant_data

    # --- Step 2: Check if all required fields are collected ---
    missing_field = None
    for field in required_fields:
        if field not in tenant_data or not tenant_data[field]:
            missing_field = field
            break

    if missing_field:
        # If a field is missing, prompt for that field and update state.
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                f"You are assisting a mall tenant with {operation_type}ing a {entity_type}. "
                f"To proceed, you need the following information: {', '.join(required_fields)}. "
                f"Currently, the '{missing_field}' is missing. "
                f"Please ask: 'What is the value for {missing_field}?'"
            )),
            ("human", (
                f"Tenant intends to {operation_type} a {entity_type}. Still need: '{missing_field}'."
            ))
        ])
        prompt_chain = prompt_template | llm | output_parser
        prompt_message = prompt_chain.invoke({
            "operation_type": operation_type,
            "entity_type": entity_type,
            "required_fields": required_fields,
            "field": missing_field
        })
        print(f"Prompting for missing field '{missing_field}': {prompt_message}")

        # Update state with the current tenant_data and mark the waiting field.
        updated_state = state.copy()
        updated_state["agent_response"] = prompt_message
        updated_state["next_node"] = "llm_call_node"
        updated_state["awaiting_tenant_input_field"] = missing_field
        updated_state["tenant_data"] = tenant_data
        return updated_state

    # --- Step 3: All required fields are collected; generate SQL query via LLM ---
    try:
        print("All required tenant data collected. Generating SQL query using LLM...")
        sql_generation_template = ChatPromptTemplate.from_messages([
            ("system", (
                f"You are a SQL query generator for a mall management system. "
                f"Generate a valid SQL query for a {operation_type} operation on {entity_type} data in a PostgreSQL database. "
                f"Use the following tenant data to construct the query: {tenant_data}. "
                f"Return only the SQL query without any additional text."
            )),
            ("human", "Generate SQL query.")
        ])
        # Invoke the chain to generate the SQL query.
        sql_query = (sql_generation_template | llm | output_parser).invoke({
            "tenant_data": tenant_data,
            "operation_type": operation_type,
            "entity_type": entity_type
        })
        print(f"Generated SQL Query: {sql_query}")

        # --- Step 4: Execute the SQL Query using the SQL tool ---
        sql_tool = SQLDatabaseTool()  # our SQL execution tool
        sql_output = sql_tool.run(sql_query)
        print(f"SQL Tool Output: {sql_output}")

        # --- Step 5: Confirm the operation ---
        confirmation_template = ChatPromptTemplate.from_messages([
            ("system", (
                f"You are a chatbot confirming a tenant's {operation_type} operation on {entity_type} data. "
                f"Provide a concise confirmation message that the operation was successful."
            )),
            ("human", f"Confirm {operation_type} operation on {entity_type}.")
        ])
        confirmation_chain = confirmation_template | llm | output_parser
        confirmation_message = confirmation_chain.invoke({
            "operation_type": operation_type,
            "entity_type": entity_type
        })

        updated_state = state.copy()
        updated_state["agent_response"] = confirmation_message
        updated_state["next_node"] = "llm_call_node"
        updated_state["tenant_data"] = {}  # Clear tenant_data after operation
        return updated_state

    except Exception as e:
        error_message = f"Error performing tenant data operation: {e}"
        print(error_message)
        error_template = ChatPromptTemplate.from_messages([
            ("system", (
                f"You are a chatbot handling errors during a tenant's {operation_type} operation on {entity_type} data. "
                f"Inform the tenant about the error concisely and advise them to try again or contact support."
            )),
            ("human", f"Error during {operation_type} operation on {entity_type}. Details: {error_message}")
        ])
        error_chain = error_template | llm | output_parser
        error_response = error_chain.invoke({
            "operation_type": operation_type,
            "entity_type": entity_type,
            "error_message": error_message
        })

        updated_state = state.copy()
        updated_state["agent_response"] = error_response
        updated_state["next_node"] = "llm_call_node"
        return updated_state
