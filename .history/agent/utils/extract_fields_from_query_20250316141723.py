from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from config.config_loader import load_config
from utils.database_utils import load_database_schema_from_cache

config = load_config()
llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
# Replace string output parser with JSON output parser
json_output_parser = JsonOutputParser()
db_schema = load_database_schema_from_cache()

# --- Enhanced Prompt Template for Field Extraction (Function i) ---
field_extraction_prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"""
        You are an expert at extracting structured information from user queries related to tenant data operations in a mall database.
        Your task is to identify and extract relevant fields and their values from the user's query for a specific entity type.
        
        **Entity Type:** {{entity_type}}
        **User Query:** {{user_query}}
        
        **IMPORTANT: You must ONLY output a valid JSON object with no additional text or explanations.**
        
        **Instructions:**
        - Analyze the User Query and identify any field names and their corresponding values that are related to the {{entity_type}}.
        - Assume that field names might be mentioned explicitly or implicitly in the query.
        - Only include fields where values can be reliably extracted.
        - Format your entire response as a valid JSON object.
        - Do not include any wrapper text, explanations, or non-JSON content.
        
        **JSON Structure Format:**
        ```json
        {{
            "field_name1": "value1",
            "field_name2": "value2"
        }}
        ```
        
        If no fields can be extracted, return an empty JSON object: {{}}
        """),
    ("human", "Extract fields and values from the user query for {entity_type} as JSON only.")
])

def extract_fields_from_query(user_query: str, entity_type: str) -> Dict[str, Any]:
    """
    (Function i)
    Extracts fields and their values from the user query using an LLM with guaranteed JSON output.
    Args:
        user_query (str): The user's natural language query.
        entity_type (str): The entity type (e.g., "offer", "store").
    Returns:
        Dict[str, Any]: A dictionary containing extracted field-value pairs.
                       Returns an empty dictionary if no fields are extracted.
    """
    print(f"extract_fields_from_query - User Query: {user_query}, Entity Type: {entity_type}")
    try:
        # Add formatting instructions and use the JSON output parser
        field_extraction_chain = field_extraction_prompt_template | llm | json_output_parser
        
        # Include response format instructions
        extraction_response = field_extraction_chain.invoke({
            "user_query": user_query,
            "entity_type": entity_type
        })
        
        print(f"extract_fields_from_query - Extracted Tenant Data (Dictionary): {extraction_response}")
        return extraction_response
        
    except Exception as e:
        print(f"extract_fields_from_query - Error during field extraction: {e}")
        # Add more specific error handling
        if "JSON" in str(e):
            print("JSON parsing error - LLM returned non-JSON format")
        return {} # Return empty dict on error

# Add this function as a fallback mechanism
def ensure_json_output(llm_response: str) -> Dict[str, Any]:
    """
    Ensures the LLM response is valid JSON by attempting multiple parsing strategies.
    Args:
        llm_response (str): The raw response from the LLM.
    Returns:
        Dict[str, Any]: A dictionary parsed from the response or empty dict if parsing fails.
    """
    import json
    import re
    
    # Try direct JSON parsing first
    try:
        return json.loads(llm_response)
    except json.JSONDecodeError:
        pass
    
    # If direct parsing fails, try to extract JSON with regex
    try:
        # Find content between curly braces including nested structures
        json_pattern = r'\{(?:[^{}]|(?R))*\}'
        match = re.search(json_pattern, llm_response)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except (json.JSONDecodeError, re.error):
        pass
    
    # Last resort: Try to extract key-value pairs from text
    try:
        # Look for patterns like "key": "value" or "key": value
        pairs = re.findall(r'"([^"]+)"\s*:\s*(?:"([^"]*)"|(true|false|\d+))', llm_response)
        if pairs:
            result = {}
            for key, str_val, non_str_val in pairs:
                result[key] = str_val if str_val else non_str_val
            return result
    except Exception:
        pass
    
    # Return empty dict if all parsing attempts fail
    return {}

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