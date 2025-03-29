import psycopg2
from psycopg2 import OperationalError, errorcodes
from config.config_loader import load_config
from langchain_community.utilities import SQLDatabase
import os
import json
from pinecone import Pinecone  # Import Pinecone

config = load_config()

def get_db_connection():
    """Get a connection to the database from the URI."""
    db_uri = config.database.db_uri
    try:
        conn = psycopg2.connect(db_uri)
        print("Connected to database.")
        return conn
    except OperationalError as e:
        print(f"Error connecting to database: {e}")
        raise e

def db_execute(query: str, params: tuple = ()):
    """Execute an SQL statement (INSERT, UPDATE, DELETE) and commit the transaction."""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        print("SQL statement executed successfully.")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error executing SQL statement: {e}")
        raise e
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def db_fetch_one(query: str, params: tuple = ()):
    """Fetch a single row from the database."""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        if row:
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))
        return None
    except Exception as e:
        print(f"Error fetching row: {e}")
        raise e
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def get_db_schema_description(conn):
    """Retrieve the schema description from the database."""
    schema_description = "Database Schema: \n\n"
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name, column_name;")
            columns_data = cur.fetchall()
            current_table = None
            for row in columns_data:
                table_name, column_name, data_type = row
                if table_name != current_table:
                    schema_description += f"\nTable: {table_name}\n"
                    current_table = table_name
                schema_description += f"{column_name}: {data_type}\n"
    except OperationalError as e:
        print(f"Error retrieving schema description: {e}")
        raise e
    return schema_description

SCHEMA_CACHE_FILE = "db_schema_cache.json"
SCHEMA_CACHE_PATH = os.path.join("data", SCHEMA_CACHE_FILE)

def fetch_database_schema_json() -> str:
    """Fetches database schema information as a JSON string using Langchain SQLDatabase."""
    db_uri = config.database.db_uri
    db = SQLDatabase.from_uri(db_uri)
    db_schema_string = db.get_table_info()
    return db_schema_string

def cache_database_schema(schema_json_string: str):
    """Caches the database schema information to a JSON file."""
    os.makedirs(os.path.dirname(SCHEMA_CACHE_PATH), exist_ok=True)
    try:
        with open(SCHEMA_CACHE_PATH, "w") as f:
            json.dump({"schema_info": schema_json_string}, f, indent=4)
        print(f"Database schema cached to: {SCHEMA_CACHE_PATH}")
    except Exception as e:
        print(f"Error caching database schema to file: {e}")

def load_database_schema_from_cache() -> str:
    """Loads the database schema information from the cache JSON file."""
    if not os.path.exists(SCHEMA_CACHE_PATH):
        print("Schema cache file not found.")
        return None
    try:
        with open(SCHEMA_CACHE_PATH, "r") as f:
            cached_data = json.load(f)
            schema_json_string = cached_data.get("schema_info")
            if schema_json_string:
                print(f"Database schema loaded from cache: {SCHEMA_CACHE_PATH}")
                return schema_json_string
            else:
                print("Schema information not found in cache file.")
                return None
    except Exception as e:
        print(f"Error loading database schema from cache file: {e}")
        return None

def get_database_schema_string(use_cache: bool = True) -> str:
    """Retrieves database schema information."""
    if use_cache:
        cached_schema = load_database_schema_from_cache()
        if cached_schema:
            return cached_schema
    print("Fetching database schema from Neon Postgres...")
    schema_string = fetch_database_schema_json()
    if schema_string:
        cache_database_schema(schema_string)
        return schema_string
    else:
        print("Error: Could not fetch database schema from Neon Postgres.")
        return ""

def get_vector_db_client():
    """Initialize and return the Pinecone client."""
    api_key = config.pineconedb.api_key
    try:
        client = Pinecone(api_key=api_key)
        print("Pinecone client initialized.")
        return client
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        raise e

def get_vector_db_index(client, index_name=None):
    """Get the Pinecone index."""
    if not index_name:
        index_name = config.pineconedb.index_name  # Default from config
    try:
        index = client.Index(index_name)
        print(f"Accessed Pinecone index: {index_name}")
        return index
    except Exception as e:
        print(f"Error accessing Pinecone index {index_name}: {e}")
        return None

def get_vector_db_description(index, limit=5):
    """Retrieve description of the Pinecone index."""
    description = "Pinecone Index Description: \n\n"
    try:
        stats = index.describe_index_stats()
        description += f"Index Stats: {stats}\n"
        # Note: Pinecone doesn't provide a direct 'peek' like ChromaDB, so stats are used here.
        # To fetch sample data, you’d need specific IDs or a query, which requires additional logic.
    except Exception as e:
        description += f"Error retrieving Pinecone index description: {e}"
    return description

def db_fetch_all(query: str, params: tuple = ()):
    """Fetch all rows from the database."""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        print(f"Error fetching all rows: {e}")
        return []
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    try:
        neon_conn = get_db_connection()
        schema_desc = get_db_schema_description(neon_conn)
        print("\n--- Database Schema Description ---")
        print(schema_desc)
        neon_conn.close()

        pinecone_client = get_vector_db_client()
        vector_db_index = get_vector_db_index(pinecone_client)
        vector_desc = get_vector_db_description(vector_db_index)
        print("\n--- Vector Database Description ---")
        print(vector_desc)
    except Exception as e:
        print(f"An error occurred during testing: {e}")