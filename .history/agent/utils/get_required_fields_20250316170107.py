from agent.nodes.intent_router_node import IntentCategory
from typing import List, Dict, Any
import re
import json

def parse_sql_schema(schema_sql: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse SQL CREATE TABLE statements into a structured dictionary.
    
    Args:
        schema_sql (str): SQL schema as CREATE TABLE statements
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping table names to their column information
    """
    tables = {}
    
    # Extract CREATE TABLE blocks
    table_blocks = re.findall(r'CREATE TABLE (\w+) \(([\s\S]*?)\)(?:;|\n\n)', schema_sql)
    
    for table_name, columns_text in table_blocks:
        tables[table_name.lower()] = {"columns": {}}
        
        # Extract column definitions
        column_defs = columns_text.strip().split(',\n\t')
        
        for column_def in column_defs:
            column_def = column_def.strip()
            
            # Skip constraints
            if column_def.startswith('CONSTRAINT'):
                continue
                
            # Extract column name and properties
            match = re.match(r'(\w+)\s+([\w\s\(\)]+)(?:\s+(.*))?', column_def)
            if match:
                col_name, col_type, constraints = match.groups()
                constraints = constraints or ""
                
                # Create column info dictionary
                column_info = {
                    "name": col_name,
                    "type": col_type.strip(),
                    "nullable": "NOT NULL" not in constraints,
                    "primary_key": "PRIMARY KEY" in constraints,
                    "auto_increment": "SERIAL" in col_type.upper() or "IDENTITY" in constraints.upper()
                }
                
                tables[table_name.lower()]["columns"][col_name] = column_info
                
    return tables

def calculate_required_fields(intent: IntentCategory, entity_type: str, schema_sql: str) -> List[str]:
    """
    (Function ii)
    Calculates the list of required fields for a tenant data operation based on intent and entity type.
    Uses cached schema information to determine required fields dynamically.
    
    Args:
        intent (IntentCategory): The tenant intent (e.g., TENANT_INSERT_OFFER).
        entity_type (str): The entity type (e.g., "offer", "store").
        schema_sql (str): The SQL schema as a string.
    
    Returns:
        List[str]: A list of required field names for the operation.
                  Returns an empty list if no required fields are determined or in case of errors.
    """
    print(f"calculate_required_fields - Intent: {intent}, Entity Type: {entity_type}")
    operation_type = None
    
    # Determine operation type from intent
    if str(intent) in [str(IntentCategory.TENANT_UPDATE_OFFER), str(IntentCategory.TENANT_UPDATE_STORE)]:
        operation_type = "update"
    elif str(intent) in [str(IntentCategory.TENANT_INSERT_OFFER), str(IntentCategory.TENANT_INSERT_STORE)]:
        operation_type = "insert"
    elif str(intent) in [str(IntentCategory.TENANT_DELETE_OFFER), str(IntentCategory.TENANT_DELETE_STORE)]:
        operation_type = "delete"
    else:
        print(f"calculate_required_fields - Unknown intent for tenant data operation: {intent}. Returning empty required fields.")
        return [] # Unknown intent, return no required fields
    
    try:
        # Parse SQL schema into structured format
        parsed_schema = parse_sql_schema(schema_sql)
        print(json(parsed_schema))
        
        # Map entity type to table name (assuming plural table names)
        table_name = entity_type + "s"  # e.g., "offer" -> "offers"
        
        # Check if table exists in the schema
        if table_name in parsed_schema:
            table_schema = parsed_schema[table_name]
            required_fields = []
            
            # Extract required fields based on operation type
            if operation_type == "insert":
                # For inserts, we need all non-nullable fields except auto-increment columns
                required_fields = [
                    col_info["name"] for col_name, col_info in table_schema["columns"].items()
                    if not col_info["nullable"] and not col_info["auto_increment"]
                ]
            elif operation_type in ["update", "delete"]:
                # For updates and deletes, we need the primary key
                required_fields = [
                    col_info["name"] for col_name, col_info in table_schema["columns"].items()
                    if col_info["primary_key"]
                ]
                
                # For updates, typically we also need at least one field to update
                if operation_type == "update":
                    # Add some common fields that might be updated (adjust based on your actual schema)
                    common_update_fields = []
                    for field in ["name", "title", "description", "price", "discount", "status"]:
                        for col_name in table_schema["columns"]:
                            if field in col_name.lower():
                                common_update_fields.append(col_name)
                                break
                    
                    # We'll suggest these fields but don't make them required
                    print(f"calculate_required_fields - Common fields that might be updated: {common_update_fields}")
            
            print(f"calculate_required_fields - Operation Type: {operation_type}, Entity: {entity_type}, Required Fields: {required_fields}")
            return required_fields
        else:
            print(f"calculate_required_fields - Schema not found for table: {table_name}. Returning empty required fields.")
            return []  # Table not found in schema
            
    except Exception as e:
        print(f"calculate_required_fields - Error processing schema: {e}. Returning empty required fields.")
        return []  # Error accessing/processing schema

# Example usage for testing
if __name__ == "__main__":
    # Import the necessary modules only in the main block
    from agent.utils.database_utils import load_database_schema_from_cache
    
    # Define a sample schema for testing
    sample_schema = """
CREATE TABLE offers (
    offer_id SERIAL NOT NULL, 
    tenant_id INTEGER NOT NULL, 
    offer_name VARCHAR(100) NOT NULL, 
    description TEXT, 
    discount_percentage NUMERIC(5, 2) NOT NULL, 
    start_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    end_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    status VARCHAR(20) DEFAULT 'active',
    CONSTRAINT offers_pkey PRIMARY KEY (offer_id),
    CONSTRAINT offers_tenant_id_fkey FOREIGN KEY(tenant_id) REFERENCES tenant_users (user_id)
)

CREATE TABLE stores (
    store_id SERIAL NOT NULL, 
    tenant_id INTEGER NOT NULL, 
    store_name VARCHAR(100) NOT NULL, 
    location VARCHAR(50) NOT NULL, 
    floor VARCHAR(20) NOT NULL, 
    category VARCHAR(50), 
    operating_hours VARCHAR(100),
    contact_number VARCHAR(20),
    CONSTRAINT stores_pkey PRIMARY KEY (store_id),
    CONSTRAINT stores_tenant_id_fkey FOREIGN KEY(tenant_id) REFERENCES tenant_users (user_id)
)
"""

    # Try to load schema from cache, if it fails, use the sample schema
    try:
        raw_schema = load_database_schema_from_cache()
        if isinstance(raw_schema, dict) and "schema_info" in raw_schema:
            schema_sql = raw_schema["schema_info"]
        else:
            schema_sql = str(raw_schema)
    except Exception as e:
        print(f"Error loading schema from cache: {e}. Using sample schema.")
        schema_sql = sample_schema
    
    print("===== TESTING REQUIRED FIELDS CALCULATION =====")
    
    # Get entity types from schema
    parsed_schema = parse_sql_schema(schema_sql)
    entity_types = [table_name[:-1] for table_name in parsed_schema.keys()]  # Remove 's' from table names
    
    print(f"Detected entity types: {entity_types}")
    
    # Test with all intents and entities
    for intent_name in dir(IntentCategory):
        # Skip non-intent attributes
        if intent_name.startswith('_') or not intent_name.startswith('TENANT_'):
            continue
        
        intent = getattr(IntentCategory, intent_name)
        
        # Find matching entity type
        for entity_type in entity_types:
            # Check if this intent applies to this entity type
            if entity_type.upper() in intent_name:
                required_fields = calculate_required_fields(intent, entity_type, schema_sql)
                print(f"Intent: {intent_name}, Entity Type: {entity_type}")
                print(f"Required Fields: {required_fields}")
                print("-" * 50)