from typing import List, Dict, Any
from utils.database_utils import load_database_schema_from_cache

class SQLTenantDataToolkit:
    """
    Toolkit for accessing and retrieving tenant database schema information.
    Uses the cached database schema loaded from utils.database_utils.
    """

    def __init__(self):
        """
        Initializes the SQLTenantDataToolkit by loading the database schema from cache.
        """
        self.db_schema = load_database_schema_from_cache()
        if not self.db_schema:
            raise ValueError("Database schema could not be loaded from cache. Please ensure the schema cache is populated.")
        self.schema_info_dict = self._build_schema_info_dict() # Build schema info dict on initialization


    def get_table_schema(self, table_name: str) -> str:
        """
        Retrieves the CREATE TABLE schema for a given table name as a string.

        Args:
            table_name (str): The name of the table (e.g., "offers", "stores").

        Returns:
            str: The CREATE TABLE schema for the table, or an empty string if not found.
        """
        if table_name in self.db_schema:
            return self.db_schema[table_name]
        else:
            return "" # Or raise an exception if you prefer to be stricter


    def get_schema_info_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing structured schema information for all tables.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are table names and values are dictionaries
                                        containing 'columns' (list of column info dicts) and potentially 'foreign_keys', etc.
        Example:
        {
            "offers": {
                "columns": [
                    {"name": "offer_id", "type": "INTEGER", "primary_key": True, "nullable": False},
                    {"name": "offer_name", "type": "VARCHAR", "primary_key": False, "nullable": False},
                    ...
                ],
                "primary_key_columns": ["offer_id"],
                "non_primary_key_columns": ["offer_name", "description", ...]
            },
            "stores": { ... },
            ...
        }
        """
        return self.schema_info_dict


    def _build_schema_info_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Builds a structured schema information dictionary from the raw CREATE TABLE schema strings.
        This is called internally during toolkit initialization.

        Returns:
            Dict[str, Dict[str, Any]]: The structured schema information dictionary.
        """
        schema_info = {}
        for table_name, create_table_statement in self.db_schema.items():
            table_info = {"columns": []}
            lines = create_table_statement.strip().split('\n')
            column_definitions_start = False
            for line in lines:
                line = line.strip().replace(",", "") # Clean up lines
                if '(' in line and not column_definitions_start:
                    column_definitions_start = True # Start processing column definitions
                    continue # Skip the line with table name itself
                if ')' in line: # End of column definitions
                    break # Stop processing when closing parenthesis is found

                if column_definitions_start and line:
                    parts = line.split() # Split line into parts
                    if len(parts) >= 2:
                        column_name = parts[0].strip('`') # Extract column name and remove backticks
                        column_type = parts[1].upper()     # Extract column type and uppercase it

                        column_info_dict = {"name": column_name, "type": column_type, "primary_key": False, "nullable": True} # Defaults

                        # Check for PRIMARY KEY and NOT NULL constraints (very basic parsing)
                        if "PRIMARY" in parts and "KEY" in parts:
                            column_info_dict["primary_key"] = True
                        if "NOT" in parts and "NULL" in parts:
                            column_info_dict["nullable"] = False

                        table_info["columns"].append(column_info_dict)
            schema_info[table_name] = table_info

        # --- Post-process to categorize columns for easier access ---
        for table_name, table_data in schema_info.items():
            table_data["primary_key_columns"] = [col["name"] for col in table_data["columns"] if col["primary_key"]]
            table_data["non_primary_key_columns"] = [col["name"] for col in table_data["columns"] if not col["primary_key"]]


        return schema_info


if __name__ == '__main__':
    # --- Example Usage and Testing ---
    toolkit = SQLTenantDataToolkit()

    # Get schema for 'offers' table
    offers_schema = toolkit.get_table_schema("offers")
    print("Schema for 'offers' table:\n", offers_schema)

    # Get schema info dictionary
    schema_info = toolkit.get_schema_info_dict()
    print("\nSchema Info Dictionary (first 2 tables):\n", dict(list(schema_info.items())[:2])) # Print info for first 2 tables for brevity

    if "offers" in schema_info:
        offer_columns_info = schema_info["offers"]["columns"]
        print("\nColumn info for 'offers' table:")
        for col_info in offer_columns_info:
            print(f"  {col_info}")

        primary_key_cols = schema_info["offers"]["primary_key_columns"]
        print("\nPrimary Key Columns for 'offers' table:", primary_key_cols)

        non_pk_cols = schema_info["offers"]["non_primary_key_columns"]
        print("\nNon-Primary Key Columns for 'offers' table:", non_pk_cols)