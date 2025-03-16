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
        Robust parsing to handle comments and variations in CREATE TABLE syntax.
        """
        schema_info = {}
        current_table_name = None
        create_table_statement_lines = []

        for line in self.db_schema["schema_info"].splitlines(): # Split the entire schema string into lines
            line = line.strip()

            if not line or line.startswith('--') or line.startswith('/*'): # Skip empty lines and comments
                continue
            if line.startswith('*/'): # Skip end of block comments
                continue

            if line.upper().startswith("CREATE TABLE"):
                current_table_name_start_index = line.upper().find("TABLE") + len("TABLE") + 1 # Find table name start
                current_table_name_end_index = line.find("(", current_table_name_start_index) # Find table name end before parenthesis
                if current_table_name_end_index == -1: # Handle cases where '(' is on the next line
                    current_table_name = line[current_table_name_start_index:].strip().strip('`').strip()
                else:
                    current_table_name = line[current_table_name_start_index:current_table_name_end_index].strip().strip('`').strip()

                if current_table_name:
                    schema_info[current_table_name] = {"columns": []}
                    create_table_statement_lines = [] # Start collecting lines for this table
                    create_table_statement_lines.append(line) # Add CREATE TABLE line
                continue # Move to the next line after processing CREATE TABLE

            if current_table_name: # If we are within a CREATE TABLE block
                create_table_statement_lines.append(line) # Collect all lines within CREATE TABLE

                if ')' in line: # Check for end of CREATE TABLE block
                    create_table_statement = "\n".join(create_table_statement_lines) # Reconstruct the full CREATE TABLE statement
                    table_info = {"columns": []}
                    column_definitions_start = False
                    for col_line in create_table_statement.splitlines(): # Process each line of CREATE TABLE statement
                        col_line = col_line.strip().replace(",", "")
                        if '(' in col_line and not column_definitions_start:
                            column_definitions_start = True
                            continue
                        if ')' in col_line:
                            break

                        if column_definitions_start and col_line:
                            parts = col_line.split()
                            if len(parts) >= 2:
                                column_name = parts[0].strip('`')
                                column_type = parts[1].upper()

                                column_info_dict = {"name": column_name, "type": column_type, "primary_key": False, "nullable": True}

                                if "PRIMARY" in parts and "KEY" in parts:
                                    column_info_dict["primary_key"] = True
                                if "NOT" in parts and "NULL" in parts:
                                    column_info_dict["nullable"] = False
                                table_info["columns"].append(column_info_dict)
                    schema_info[current_table_name] = table_info # Update schema info with parsed columns
                    current_table_name = None # Reset for next table


        # --- Post-process to categorize columns (same as before) ---
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