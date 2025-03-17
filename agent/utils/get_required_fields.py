from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Optional, Union, Any
import json
import ast
import enum
import time
from config.config_loader import load_config

config = load_config()
# Assuming IntentCategory is defined as shown
class IntentCategory(enum.Enum):
    TENANT_INSERT_STORE = "TENANT_INSERT_STORE"
    TENANT_UPDATE_STORE = "TENANT_UPDATE_STORE"
    TENANT_DELETE_STORE = "TENANT_DELETE_STORE"
    TENANT_INSERT_OFFER = "TENANT_INSERT_OFFER"
    TENANT_UPDATE_OFFER = "TENANT_UPDATE_OFFER"
    TENANT_DELETE_OFFER = "TENANT_DELETE_OFFER"
    TENANT_QUERY = "TENANT_QUERY"

# Enhanced prompt template with more explicit instructions and format requirements
required_fields_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are an expert at determining the required fields for tenant data operations in a mall database, based on the user's intent and the database schema.
        Your task is to analyze the user's intent, consider the provided database schema, and identify the fields that are absolutely necessary to perform the requested operation.

        **Database Schema:**
        {db_schema}

        **Tenant Intent:** {intent}
        **User Query:** {user_query}
        **Existing fields:** {existing_fields}

        **Instructions:**
        - Analyze the Tenant Intent to understand the type of data operation (e.g., insert, update, delete) and the entity type (e.g., offer, store).
        - Consider the User Query to see if the user has already provided any information relevant to the required fields.
        - Based on the Database Schema, determine the list of fields that are essential to successfully execute the database operation.
        - For 'insert' operations, this typically includes all NOT NULL columns that are not automatically generated (e.g., not SERIAL primary keys).
        - For 'update' operations, you need to identify the fields that can uniquely identify the record to be updated (e.g., primary key fields) PLUS at least one field that will be updated.
        - For 'delete' operations, you only need to identify the fields that can uniquely identify the record to be deleted (e.g., primary key fields or unique identifying fields).
        - Return a list of field names that are REQUIRED to proceed with the tenant's request.
        - IMPORTANT: Return ONLY a valid Python list of strings in the exact format shown in the example below. Do not include any explanatory text.
        - Existing fields should not be present in the required fields list because they are already available for us.

        **Example Output Format (for TENANT_INSERT_OFFER intent):**
        ["store_id", "product_id", "title", "description", "discount_percentage", "start_date", "end_date"]
        """),
    ("human", "Determine the required fields for {intent} based on the user query and database schema and existing fields.")
])

class RequiredFieldsCalculator:
    """Enhanced class for calculating required fields with caching and validation"""
    
    def __init__(self, model_name=config.llm.model_name, api_key=config.llm.api_key, cache_ttl=3600):
        """
        Initialize the calculator with specified model and caching parameters.
        
        Args:
            model_name (str): The name of the LLM model to use
            model_provider (str): The provider of the LLM model
            cache_ttl (int): Time-to-live for cached results in seconds
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
        self.output_parser = StrOutputParser()
        self.cache = {}
        self.cache_ttl = cache_ttl
        
    def _get_cache_key(self, intent: IntentCategory, user_query: str, db_schema: str) -> str:
        """Generate a cache key from the input parameters"""
        return f"{intent.value}|{user_query}|{hash(db_schema)}"
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid based on TTL"""
        return (time.time() - cache_entry.get('timestamp', 0)) < self.cache_ttl
    
    def _validate_fields(self, fields: List[str], db_schema: str) -> List[str]:
        """
        Validate that the fields exist in the database schema.
        This is a simple implementation - in production you might want to parse the schema properly.
        
        Args:
            fields (List[str]): List of field names to validate
            db_schema (str): Database schema string
            
        Returns:
            List[str]: Validated list of fields (removing any that don't exist in schema)
        """
        # Simple validation - check if field name appears in schema
        # In a real implementation, you'd parse the schema and check properly
        validated_fields = []
        for field in fields:
            # This is simplified - in reality you'd want to match field names more precisely
            if field.strip() in db_schema:
                validated_fields.append(field.strip())
        return validated_fields
    
    def calculate_required_fields(self, intent: IntentCategory, user_query: str, db_schema: str, existing_fields: str) -> List[str]:
        """
        Calculates the list of required fields for a tenant action based on intent, user query, and database schema using an LLM.
        Uses caching to improve performance for repeated queries.

        Args:
            intent (IntentCategory): The identified tenant intent.
            user_query (str): The user's natural language query.
            db_schema (str): The database schema string.

        Returns:
            List[str]: A list of required field names (strings).
                      Returns an empty list if required fields cannot be determined or in case of error.
        """
        
        # Check cache first
        cache_key = self._get_cache_key(intent, user_query, db_schema)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print(f"calculate_required_fields - Using cached result")
            return self.cache[cache_key]['fields']
        
        try:
            # Create and invoke the LangChain chain
            required_fields_chain = required_fields_prompt_template | self.llm | self.output_parser
            required_fields_response_str = required_fields_chain.invoke({
                "intent": intent.value,  # Pass intent as string value
                "user_query": user_query,
                "db_schema": db_schema,
                "existing_fields": existing_fields
            })

            # Parse LLM Response to List using multiple approaches
            required_fields_list = self._parse_llm_response(required_fields_response_str)
            
            if not required_fields_list:
                return []
            
            # Validate fields against schema
            validated_fields = self._validate_fields(required_fields_list, db_schema)
            
            # Cache the result
            self.cache[cache_key] = {
                'fields': validated_fields,
                'timestamp': time.time()
            }
            
            return validated_fields

        except Exception as e:
            print(f"calculate_required_fields - Error during required fields calculation: {e}")
            return []  # Return empty list on error
    
    def _parse_llm_response(self, response_str: str) -> List[str]:
        """
        Attempts to parse the LLM response string into a list using multiple methods
        
        Args:
            response_str (str): The raw string response from the LLM
            
        Returns:
            List[str]: The parsed list of fields, or empty list if parsing fails
        """
        # Clean up the response string
        response_str = response_str.strip()
        
        # Try different parsing methods in order of preference
        parsing_methods = [
            self._parse_with_ast,
            self._parse_with_json,
            self._parse_with_string_manipulation
        ]
        
        for method in parsing_methods:
            try:
                result = method(response_str)
                if result and isinstance(result, list) and all(isinstance(item, str) for item in result):
                    return result
            except Exception as e:
                print(f"Parsing method {method.__name__} failed: {e}")
                continue
                
        return []
    
    def _parse_with_ast(self, response_str: str) -> List[str]:
        """Parse using Python's ast module"""
        return ast.literal_eval(response_str)
    
    def _parse_with_json(self, response_str: str) -> List[str]:
        """Parse using JSON parser"""
        return json.loads(response_str)
    
    def _parse_with_string_manipulation(self, response_str: str) -> List[str]:
        """Parse using string manipulation for basic list formats"""
        if response_str.startswith('[') and response_str.endswith(']'):
            # Remove brackets and split by commas
            content = response_str[1:-1]
            # Handle quoted strings
            if '"' in content:
                items = [item.strip().strip('"') for item in content.split('",')]
            elif "'" in content:
                items = [item.strip().strip("'") for item in content.split("',")]
            else:
                items = [item.strip() for item in content.split(',')]
            
            # Remove empty strings and trailing quotes
            items = [item.rstrip('"\'') for item in items if item]
            return items
        return []

from utils.database_utils import get_db_schema_description, get_db_connection
conn = get_db_connection()
# Test database schema
TEST_DB_SCHEMA = get_db_schema_description(conn)


def main():
    """Main function to demonstrate the RequiredFieldsCalculator"""
    calculator = RequiredFieldsCalculator()
    
    try:
        while True:
            print("\nSelect an intent:")
            for i, intent in enumerate(IntentCategory):
                print(f"{i+1}. {intent.value}")
            
            intent_choice = int(input("Enter intent number: ")) - 1
            if intent_choice < 0 or intent_choice >= len(IntentCategory):
                print("Invalid choice. Please try again.")
                continue
                
            selected_intent = list(IntentCategory)[intent_choice]
            user_query = input("Enter user query: ")
            existing_fields = input("Enter fields you already have")
            
            result = calculator.calculate_required_fields(selected_intent, user_query, TEST_DB_SCHEMA, existing_fields=existing_fields)
            print(f"\nRequired fields for {selected_intent.value}:")
            print(result)
            
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
    
if __name__ == "__main__":
    main()