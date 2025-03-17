from typing import Dict, Any, List
import re
from langchain.tools import BaseTool
from agent.utils.database_utils import get_db_connection
from agent.tools.sql_tool import SQLDatabaseTool  # Import your SQL execution tool

class PrimaryKeyHandler:
    """
    Handles the identification and auto-population of primary keys in tenant operations.
    """
    
    def __init__(self, db_schema_str: str, sql_tool: SQLDatabaseTool):
        self.db_schema_str = db_schema_str
        self.sql_tool = sql_tool
        self.primary_keys = {
            "offers": "offer_id",
            "stores": "store_id",
            "events": "event_id"
        }
    
    def is_primary_key(self, field: str, entity_type: str) -> bool:
        """Check if a field is a primary key for the given entity type."""
        table_mapping = {
            "offer": "offers",
            "store": "stores",
            "event": "events",
        }
        
        table_name = table_mapping.get(entity_type, entity_type)
        
        return (field.lower() == f"{entity_type}_id" or 
                (table_name in self.primary_keys and field.lower() == self.primary_keys[table_name]))
    
    def handle_primary_keys(self, required_fields: List[str], entity_type: str, 
                           operation_type: str, tenant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle primary key fields in the required fields list:
        - For inserts: Mark them for auto-generation or DB default
        - For updates/deletes: Look up based on other identifying information
        
        Returns: Updated tenant_data with primary keys handled
        """
        updated_tenant_data = tenant_data.copy()
        
        for field in required_fields:
            if self.is_primary_key(field, entity_type):
                if operation_type == "insert":
                    updated_tenant_data[field] = "__AUTO_GENERATED__"
                    print(f"Primary key {field} marked for auto-generation")
                    
                elif operation_type in ["update", "delete"]:
                    if field not in updated_tenant_data:
                        lookup_id = self._lookup_entity_id(entity_type, updated_tenant_data)
                        print("LOOK UP ID: \n", lookup_id)
                        if lookup_id:
                            updated_tenant_data[field] = lookup_id
                            print(f"Primary key {field} looked up with value {lookup_id}")
                        else:
                            updated_tenant_data[field] = "__NEED_LOOKUP__"
                            print(f"Primary key {field} needs lookup with more information")
        
        return updated_tenant_data
    
    def _lookup_entity_id(self, entity_type: str, tenant_data: Dict[str, Any]) -> str:
        """
        Look up entity ID based on other identifying information using SQLDatabaseTool.
        
        Returns: Entity ID if found, otherwise None
        """
        print("ENTITY TYPE AND DATA RECEIVED BY LOOKUP FUNC: \n", entity_type, " ", tenant_data)
        unique_identifiers = {
            "offer": ("offers", "name"),
            "store": ("stores", "name"),
            "event": ("events", "name"),
        }
        
        primary_keys = {
            "offers": "offer_id",
            "stores": "store_id",
            "events": "event_id",
        }
        
        if entity_type not in unique_identifiers:
            return None
        
        table_name, identifier_field = unique_identifiers[entity_type]
        
        if identifier_field in tenant_data:
            query = f"SELECT {self.primary_keys[table_name]} FROM {table_name} WHERE {identifier_field} = '{tenant_data[identifier_field]}' LIMIT 1;"
            print(f"Executing query: {query}")
            
            result = self.sql_tool._run(self, query=query)  # Execute SQL query using SQLDatabaseTool
            
            if "No results found" in result or "Error" in result:
                return None
            
            try:
                # Extracting ID from SQLDatabaseTool output
                lines = result.split("\n")
                if len(lines) > 2:  # Check if there are results
                    entity_id = lines[2].strip("| ").split(" | ")[0]  # Extract ID
                    return entity_id
            except Exception as e:
                print(f"Error parsing SQL result: {e}")
        
        return None
