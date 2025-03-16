from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
# # from config.config_loader import load_config
# from utils.database_utils import load_database_schema_from_cache

# config = load_config()
llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
output_parser = StrOutputParser()
# db_schema_str = load_database_schema_from_cache()

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


def extract_fields_from_query(user_query: str, entity_type: str) -> Dict[str, Any]:
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
    
if __name__ == '__main__':
    # --- Test extract_fields_from_query ---
    test_queries_offers = [
        "Update offer Summer Sale discount to 30%",
        "Set discount percentage to 50% for offer named Black Friday Deal",
        "Change the title of offer to ' новогодние скидки ' and set discount to 25%", # Russian title, mixed entities
        "I want to update offer discount to 10 percent", # Implicit title missing, only discount present
        "Update something", # Very vague query, should extract nothing or very little
        "offer update with title 'Back to School' and discount 40%" # Another variation
    ]

    print("\n--- Testing extract_fields_from_query for 'offer' entity ---")
    for query in test_queries_offers:
        extracted_fields = extract_fields_from_query(query, "offer")
        print(f"\nQuery: '{query}'")
        print(f"Extracted Fields: {extracted_fields}")


    test_queries_stores = [
        "Update store 'Style Haven' operating hours to '10 AM - 11 PM'",
        "Change operating_hours for store Food Fusion to '11:00 - 23:00'",
        "store update operating hours to '9am - 9pm'", # Implicit store name missing
        "Update store timings", # Very vague store update query
        "store operation hours to '8 AM - 10 PM' for store name 'TechTrendz'" # Another variation
    ]
    print("\n--- Testing extract_fields_from_query for 'store' entity ---")
    for query in test_queries_stores:
        extracted_fields = extract_fields_from_query(query, "store")
        print(f"\nQuery: '{query}'")
        print(f"Extracted Fields: {extracted_fields}")


