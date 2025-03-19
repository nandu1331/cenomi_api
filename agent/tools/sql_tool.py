from typing import Dict, Any
from langchain.tools import BaseTool
from agent.utils.database_utils import get_db_connection

class SQLDatabaseTool(BaseTool):
    name: str = "sql_database_query"
    description: str = "Useful for retrieving structured information from the mall database. Input should be a precise SQL query for Neon Postgres."

    def _run(self, query: str) -> str:
        connection = get_db_connection()
        if not connection:
            return "Error: Unable to connect to the database."
        
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                connection.commit()
                
                if query.strip().lower().startswith("select"):
                    results = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    
                    if not results:
                        return "No results found for your query."
                    
                    output = "SQL Database Query Results:\n\n"
                    output += "| " + " | ".join(column_names) + " |\n"
                    output += "| " + " | ".join(["---"] * len(column_names)) + " |\n"
                    for row in results:
                        output += "| " + " | ".join([str(val) for val in row]) + " |\n"
                    return output
                return "Query executed successfully."
        except Exception as e:
            return f"Error during SQL query execution: {e}"
        finally:
            connection.close()
        
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Asynchronous _arun method not implemented for SQLDatabaseTool.")