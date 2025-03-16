from langchain_core.prompts import ChatPromptTemplate
from agent.nodes.intent_router_node import IntentCategory
from typing import List
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
output_parser = StrOutputParser()

required_fields_prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"""
        You are an expert at determining the required fields for tenant data operations in a mall database, based on the user's intent and the database schema.
        Your task is to analyze the user's intent, consider the provided database schema, and identify the fields that are absolutely necessary to perform the requested operation.

        **Database Schema:**
        {{db_schema}}

        **Tenant Intent:** {{intent}}
        **User Query:** {{user_query}}

        **Instructions:**
        - Analyze the Tenant Intent to understand the type of data operation (e.g., insert, update, delete) and the entity type (e.g., offer, store).
        - Consider the User Query to see if the user has already provided any information relevant to the required fields.
        - Based on the Database Schema, determine the list of fields that are essential to successfully execute the database operation.
        - For 'insert' operations, this typically includes all NOT NULL columns that are not automatically generated (e.g., not SERIAL primary keys).
        - For 'update' and 'delete' operations, you need to identify the fields that can uniquely identify the record to be updated or deleted (e.g., primary key fields or unique identifying fields).
        - Return a list of field names that are REQUIRED to proceed with the tenant's request.
        - Only return a Python list of strings (field names). Do not include any explanatory text or sentences.

        **Example Output Format (for TENANT_INSERT_OFFER intent):**
        [
            "store_id",
            "product_id",
            "title",
            "description",
            "discount_percentage",
            "start_date",
            "end_date"
        ]
        """),
    ("human", "Determine the required fields for {intent} based on the user query and database schema.")
])


def calculate_required_fields(intent: IntentCategory, user_query: str, db_schema: str) -> List[str]:
    """
    (Function ii)
    Calculates the list of required fields for a tenant action based on intent, user query, and database schema using an LLM.

    Args:
        intent (IntentCategory): The identified tenant intent.
        user_query (str): The user's natural language query.
        db_schema (str): The database schema string.

    Returns:
        List[str]: A list of required field names (strings).
                     Returns an empty list if required fields cannot be determined or in case of error.
    """
    print(f"calculate_required_fields - Intent: {intent}, User Query: {user_query}")
    try:
        required_fields_chain = required_fields_prompt_template | llm | output_parser
        required_fields_response_str = required_fields_chain.invoke({
            "intent": intent.value, # Pass intent as string value
            "user_query": user_query,
            "db_schema": db_schema
        })
        print(f"calculate_required_fields - LLM Required Fields Response (String): {required_fields_response_str}")

        # --- Parse LLM Response to List ---
        try:
            import ast
            required_fields_list: List[str] = ast.literal_eval(required_fields_response_str) # Safely evaluate string to Python list
            if not isinstance(required_fields_list, list):
                print(f"calculate_required_fields - Error: LLM did not return a list. Response was: {required_fields_response_str}. Returning empty list.")
                return []
            print(f"calculate_required_fields - Required Fields (List): {required_fields_list}")
            return required_fields_list
        except (ValueError, SyntaxError) as e: # Catch both parsing errors
            print(f"calculate_required_fields - Error parsing LLM response to list: {e}. Response was: {required_fields_response_str}. Returning empty list.")
            return [] # Return empty list if parsing fails


    except Exception as e:
        print(f"calculate_required_fields - Error during required fields calculation: {e}")
        return [] # Return empty list on error