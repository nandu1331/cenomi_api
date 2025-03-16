from typing import Dict, Any, List
from tools.sql_tool import SQLTenantDataToolkit
from langchain_core.prompts import ChatPromptTemplate

field_extraction_prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"""
        You are an expert at extracting structured information from user queries related to tenant data operations in a mall database.
        Your task is to identify and extract relevant fields and their values from the user's query for a specific entity type.

        **Entity Type:** {{entity_type}}
        **User Query:** {{user_query}}

        **Instructions:**
        - Analyze the User Query and identify any field names and their corresponding values that are related to the {{entity_type}}.
        - Assume that field names might be mentioned explicitly or implicitly in the query.
        - Extract the field names and their values as a JSON-like dictionary.
        - If a field value is not explicitly mentioned or cannot be reliably extracted, do not include it in the output dictionary.
        - Only output a valid JSON-like dictionary. Do not include any explanatory text or sentences.

        **Example Output Format (for Offer entity):**
        {{
            "offer_name": "Summer Sale",
            "discount_percentage": "30%",
            "description": "Discount on summer clothing"
        }}
        """),
    ("human", "Extract fields and values from the user query for {entity_type}.")
])


def extract_fields_from_query(user_query: str, entity_type: str, llm, output_parser) -> Dict[str, Any]:
    """
    (Function i)
    Extracts fields and their values from the user query using an LLM.

    Args:
        user_query (str): The user's natural language query.
        entity_type (str): The entity type (e.g., "offer", "store").

    Returns:
        Dict[str, Any]: A dictionary containing extracted field-value pairs.
                       Returns an empty dictionary if no fields are extracted.
    """
    print(f"extract_fields_from_query - User Query: {user_query}, Entity Type: {entity_type}")
    try:
        field_extraction_chain = field_extraction_prompt_template | llm | output_parser
        extraction_response_str = field_extraction_chain.invoke({
            "user_query": user_query,
            "entity_type": entity_type
        })
        print(f"extract_fields_from_query - LLM Extraction Response (String): {extraction_response_str}")

        # --- Parse LLM Response to Dictionary ---
        try:
            import json
            extracted_data: Dict[str, Any] = json.loads(extraction_response_str) # Attempt to parse as JSON
            print(f"extract_fields_from_query - Extracted Tenant Data (Dictionary): {extracted_data}")
            return extracted_data
        except json.JSONDecodeError as e:
            print(f"extract_fields_from_query - JSONDecodeError: {e}. Could not parse LLM response to JSON. Returning empty dictionary.")
            return {} # Return empty dict if JSON parsing fails

    except Exception as e:
        print(f"extract_fields_from_query - Error during field extraction: {e}")
        return {} # Return empty dict on error


