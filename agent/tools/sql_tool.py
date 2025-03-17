from typing import Dict, Any
from langchain.tools import BaseTool
from utils.database_utils import get_db_connection

class SQLDatabaseTool(BaseTool):
    """Tool to execute SQL queries and retrieve results from the database."""

    name: str = "sql_database_query"
    description: str = (
        "Useful for retrieving structured information directly from the mall database. "
        "Input should be a SQL query. Be precise and use correct SQL syntax for Neon Postgres. "
        "Use this tool for queries that require structured data or specific filtering and aggregation from the database. "
        "Particularly useful for listing information, counting, or retrieving specific records based on defined criteria."
    )
    
    def _run(self, query: str) -> str:
        """Execute the SQL query and return the results."""
        print(f"--- SQLDatabaseTool: Running with query: {query} ---")
        
        connection = get_db_connection()
        if not connection:
            return "Error: Unable to connect to the database."
        
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                connection.commit()
                
                # Check if the query is a SELECT query
                if query.strip().lower().startswith("select"):
                    results = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    
                    if not results:
                        return "No results found for your query."
                    
                    output_string = "SQL Database Query Results:\n\n"
                    output_string += "| " + " | ".join(column_names) + " |\n"
                    output_string += "| " + " | ".join(["---"] * len(column_names)) + " |\n"
                    
                    for row in results:
                        row_str = "| " + " | ".join([str(val) for val in row]) + " |\n"
                        output_string += row_str
                        
                    return output_string
                else:
                    # For non-SELECT queries, just return a success message.
                    return "Query executed successfully."
        
        except Exception as e:
            error_message = f"Error during SQL query execution: {e}"
            print(error_message)
            return error_message
        finally:
            connection.close()
        
    async def _arun(self, query: str) -> str:
        """Execute the SQL query and return the results asynchronously."""
        raise NotImplementedError("Asynchronous _arun method not implemented for SQLDatabaseTool.")
