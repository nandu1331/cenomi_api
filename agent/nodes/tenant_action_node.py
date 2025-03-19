from typing import Dict, Any, List
from agent.agent_state import AgentState
from agent.nodes.intent_router_node import IntentCategory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config
from agent.utils.database_utils import get_db_connection, get_db_schema_description, load_database_schema_from_cache
from agent.utils.extract_fields_from_query import extract_fields_from_query
from agent.utils.get_required_fields import RequiredFieldsCalculator
from agent.utils.primary_key_handler import PrimaryKeyHandler
from agent.tools.sql_tool import SQLDatabaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

config = load_config()
llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key)
output_parser = StrOutputParser()
db_schema_str = load_database_schema_from_cache()
required_fields_cal = RequiredFieldsCalculator()

def tenant_action_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state["role"] != "tenant":
        return {"response": "Only tenants can perform update operations."}
    
    if not state["store_id"]:
        return {"response": "No store associated with this tenant account."}

    intent = IntentCategory(state["intent"])
    entity_type = "store" if "STORE" in intent.value else "offer"
    user_query = state["user_query"]

    conn = get_db_connection()
    schema = get_db_schema_description(conn)
    sql_tool = SQLDatabaseTool()
    pk_handler = PrimaryKeyHandler(schema, sql_tool)
    
    extracted_fields = extract_fields_from_query(user_query, entity_type)
    required_fields = required_fields_cal.calculate_required_fields(intent, user_query, schema, str(list(extracted_fields.keys())))
    missing_fields = [f for f in required_fields if f not in extracted_fields and not pk_handler.is_primary_key(f, entity_type)]

    if missing_fields:
        return {"response": f"Please provide: {', '.join(missing_fields)}"}

    operation_type = "update" if "UPDATE" in intent.value else "insert"
    tenant_data = pk_handler.handle_primary_keys(required_fields, entity_type, operation_type, extracted_fields)
    tenant_data["store_id"] = state["store_id"]

    table = "stores" if entity_type == "store" else "offers"
    if operation_type == "update":
        pk_field = pk_handler.primary_keys[table]
        set_clause = ", ".join([f"{k} = '{v}'" for k, v in tenant_data.items() if k != pk_field and k != "store_id"])
        query = f"UPDATE {table} SET {set_clause} WHERE {pk_field} = '{tenant_data[pk_field]}' AND store_id = {state['store_id']}"
    else:
        cols = ", ".join(tenant_data.keys())
        vals = ", ".join([f"'{v}'" for v in tenant_data.values()])
        query = f"INSERT INTO {table} ({cols}) VALUES ({vals})"

    result = sql_tool._run(query)
    return {"tool_outputs": [{"tool": "sql_database_query", "output": result}]}

def handle_tenant_data_operation(state: AgentState, intent: IntentCategory, user_query: str, main_query: str) -> AgentState:
    if state.get("awaiting_tenant_input_field") is None:
        state["tenant_data"] = {}
    if "tenant_data" not in state:
        state["tenant_data"] = {}
    tenant_data: Dict[str, Any] = state.get("tenant_data", {})
    entity_type = operation_type = None
    current_field_index: int = state.get("current_field_index", 0)
    
    if intent == IntentCategory.TENANT_UPDATE_OFFER:
        entity_type, operation_type = "offer", "update"
    elif intent == IntentCategory.TENANT_INSERT_OFFER:
        entity_type, operation_type = "offer", "insert"
    elif intent == IntentCategory.TENANT_DELETE_OFFER:
        entity_type, operation_type = "offer", "delete"
    elif intent == IntentCategory.TENANT_INSERT_STORE:
        entity_type, operation_type = "store", "insert"
    elif intent == IntentCategory.TENANT_UPDATE_STORE:
        entity_type, operation_type = "store", "update"
    else:
        state["next_node"] = "llm_call_node"
        return state

    extracted_data = extract_fields_from_query(user_query, entity_type)
    state["tenant_data"].update(extracted_data)
    
    if not current_field_index:
        required_fields = required_fields_cal.calculate_required_fields(intent, main_query, db_schema_str, ", ".join(extracted_data.keys()))
    
    pk_handler = PrimaryKeyHandler(db_schema_str, SQLDatabaseTool)
    state["tenant_data"] = pk_handler.handle_primary_keys(required_fields, entity_type, operation_type, state["tenant_data"])
    
    missing_fields = [field for field in required_fields if field not in state.get("tenant_data", {}) and not pk_handler.is_primary_key(field, entity_type)]
    
    while missing_fields:
        field = missing_fields[0]
        if field not in state.get("tenant_data", {}):
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", f"You are assisting a tenant to {operation_type} a {entity_type}. We need more information."),
                ("human", f"Could you please provide the **{field}** for the {operation_type} {entity_type}?")
            ])
            prompt_chain = prompt_template | llm | output_parser
            prompt_message = prompt_chain.invoke({"operation_type": operation_type, "entity_type": entity_type, "field": field})

            updated_state = state.copy()
            updated_state["agent_response"] = prompt_message
            updated_state["next_node"] = "output_node"
            updated_state["awaiting_tenant_input_field"] = field
            return updated_state
        else:
            missing_fields.pop(0)

    try:
        enhanced_user_query_prompt_template = ChatPromptTemplate.from_messages([
            ("system",
                """
                You are an expert at generating natural language user queries for a database interaction system.
                Generate a concise and natural user query string based on the Database Schema, Tenant Operation, and Tenant Data.
                Example: "Insert a new offer named 'Summer Sale' with description 'Clothing sale for summer', discount percentage 30%, start date 2025-03-15, and end date 2025-03-20 in the offers table."
                Database Schema: {db_schema}
                Tenant Operation: {operation_type} {entity_type}
                Tenant Data: {tenant_data}
                """),
            ("human", "Generate a user query string for the tenant operation.")
        ])
        
        enhanced_user_query_chain = enhanced_user_query_prompt_template | llm | output_parser
        enhanced_user_query = enhanced_user_query_chain.invoke({
            "operation_type": operation_type, "entity_type": entity_type,
            "tenant_data": tenant_data, "db_schema": db_schema_str
        })

        updated_state = state.copy()
        updated_state["user_query"] = enhanced_user_query
        updated_state["next_node"] = "tool_selection_node"
        updated_state["current_field_index"] = None
        updated_state["awaiting_tenant_input_field"] = None
        return updated_state
    except Exception as e:
        error_response_template = ChatPromptTemplate.from_messages([
            ("system", f"Chatbot error during tenant {operation_type} operation."),
            ("human", "Error during user query generation for {operation_type} operation. Details: {error_message}")
        ])
        error_chain = error_response_template | llm | output_parser
        error_response = error_chain.invoke({"operation_type": operation_type, "entity_type": entity_type, "error_message": str(e)})

        updated_state = state.copy()
        updated_state["agent_response"] = error_response
        updated_state["next_node"] = "output_node"
        updated_state["current_field_index"] = None
        updated_state["awaiting_tenant_input_field"] = None
        return updated_state