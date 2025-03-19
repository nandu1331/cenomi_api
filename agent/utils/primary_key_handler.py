from typing import Dict, Any, List
from agent.tools.sql_tool import SQLDatabaseTool

class PrimaryKeyHandler:
    def __init__(self, db_schema_str: str, sql_tool: SQLDatabaseTool):
        self.db_schema_str = db_schema_str
        self.sql_tool = sql_tool
        self.primary_keys = {"offers": "offer_id", "stores": "store_id", "events": "event_id"}
    
    def is_primary_key(self, field: str, entity_type: str) -> bool:
        table_mapping = {"offer": "offers", "store": "stores", "event": "events"}
        table_name = table_mapping.get(entity_type, entity_type)
        return (field.lower() == f"{entity_type}_id" or 
                (table_name in self.primary_keys and field.lower() == self.primary_keys[table_name]))
    
    def handle_primary_keys(self, required_fields: List[str], entity_type: str, 
                           operation_type: str, tenant_data: Dict[str, Any]) -> Dict[str, Any]:
        updated_tenant_data = tenant_data.copy()
        
        for field in required_fields:
            if self.is_primary_key(field, entity_type):
                if operation_type == "insert":
                    updated_tenant_data[field] = "__AUTO_GENERATED__"
                elif operation_type in ["update", "delete"] and field not in updated_tenant_data:
                    lookup_id = self._lookup_entity_id(entity_type, updated_tenant_data)
                    updated_tenant_data[field] = lookup_id if lookup_id else "__NEED_LOOKUP__"
        
        return updated_tenant_data
    
    def _lookup_entity_id(self, entity_type: str, tenant_data: Dict[str, Any]) -> str:
        unique_identifiers = {"offer": ("offers", "name"), "store": ("stores", "name"), "event": ("events", "name")}
        primary_keys = {"offers": "offer_id", "stores": "store_id", "events": "event_id"}
        
        if entity_type not in unique_identifiers:
            return None
        
        table_name, identifier_field = unique_identifiers[entity_type]
        if identifier_field not in tenant_data:
            return None
        
        query = f"SELECT {primary_keys[table_name]} FROM {table_name} WHERE {identifier_field} = '{tenant_data[identifier_field]}' LIMIT 1;"
        result = self.sql_tool._run(query)
        
        if "No results found" in result or "Error" in result:
            return None
        
        try:
            lines = result.split("\n")
            if len(lines) > 2:
                return lines[2].strip("| ").split(" | ")[0]
        except Exception:
            return None