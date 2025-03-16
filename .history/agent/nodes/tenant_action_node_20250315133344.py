from typing import Dict, Any, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from tools.sql_tool import SQLDatabaseTool  # This is our sql_database_query tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config

config = load_config()

def tenant_action_node(state: AgentState) -> AgentState:
    """
    Tenant Action Node: Handles tenant data manipulation (CRUD) operations.
    """
    print("--- Tenant Action Node ---")
    intent: IntentCategory = state.get("intent")
    user_query: str = state.get("user_query")

    print(f"Tenant Action Node - Intent: {intent}, User Query: {user_query}")

    updated_state: AgentState = state.copy()
    return handle_tenant_data_operation(updated_state, intent)


def handle_tenant_data_operation(state: AgentState, intent: IntentCategory) -> AgentState:
    """
    Generalized handler for tenant data operations on various entities.
    Collects required fields one by one and, once complete, uses an LLM chain to
    generate the SQL query, which is then executed via the sql_database_query tool.
    """
    print("--- handle_tenant_data_operation ---")
    tenant_data: Dict[str, Any] = state.get("tenant_data", {})
    entity_type = None
    operation_type = None
    required_fields: List[str] = []

    # Instantiate the SQL tool and LLM components.
    sql_tool = SQLDatabaseTool()  # Our dedicated SQL tool for executing queries.
    llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
    output_parser = StrOutputParser()

    # --- 1. Determine Entity and Operation Type based on Intent ---
    if intent == IntentCategory.TENANT_UPDATE_OFFER:
        entity_type = "offer"
        operation_type = "update"
        required_fields = ["offer_name", "discount_percentage"]  # Add more fields as needed.
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
        print(f"handle_tenant_data_operation - No mapping for intent: {intent}. Defaulting to LLM call node.")
        state["next_node"] = "llm_call_node"
        return state
    
    state["awaiting_tenant_input_field"] = required_fields[0]

    # --- 2. Check if waiting for a specific field from previous turn ---
    if "awaiting_tenant_input_field" in state:
        waiting_field = state["awaiting_tenant_input_field"]
        # Update the tenant_data with the tenant's answer from the current user_query.
        tenant_data[waiting_field] = state["user_query"].strip()
        print(f"Updated tenant_data[{waiting_field}]: {tenant_data[waiting_field]}")
        state.pop("awaiting_tenant_input_field", None)
        state["tenant_data"] = tenant_data

    # --- 3. Collect Missing Fields One by One ---
    for field in required_fields:
        if field not in tenant_data or not tenant_data[field]:
            # Prompt the tenant for this missing field.
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", (
                    f"You are assisting a mall tenant with {operation_type}ing a {entity_type}. "
                    f"To proceed, you need the following information: {', '.join(required_fields)}. "
                    f"Currently, the '{field}' is missing. Please ask: 'What is the value for {field}?'"
                )),
                ("human", (
                    f"Tenant intends to {operation_type} a {entity_type}. Still need: '{field}'."
                ))
            ])
            prompt_chain = prompt_template | llm | output_parser
            prompt_message = prompt_chain.invoke({
                "operation_type": operation_type,
                "entity_type": entity_type,
                "required_fields": required_fields,
                "field": field
            })
            print(f"Prompting for missing field '{field}': {prompt_message}")

            updated_state = state.copy()
            updated_state["agent_response"] = prompt_message
            updated_state["next_node"] = "llm_call_node"
            updated_state["awaiting_tenant_input_field"] = field
            updated_state["tenant_data"] = tenant_data
            return updated_state

    # --- 4. All required fields collected: Generate SQL Query via LLM ---
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
        # Format the prompt with tenant_data converted to string.
        formatted_sql_prompt = sql_generation_template.format()
        # Invoke the chain to generate the SQL query.
        sql_query = (sql_generation_template | llm | output_parser).invoke({
            "tenant_data": tenant_data,
            "operation_type": operation_type,
            "entity_type": entity_type
        })
        print(f"Generated SQL Query: {sql_query}")

        # --- 5. Execute the SQL Query using the SQL tool ---
        sql_output = sql_tool.run(sql_query)
        print(f"SQL Tool Output: {sql_output}")

        # --- 6. Confirm the operation to the tenant ---
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
