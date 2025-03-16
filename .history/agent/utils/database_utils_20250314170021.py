import psycopg2
from psycopg2 import OperationalError, errorcodes
from config.config_loader import load_config
import chromadb
from langchain_community.utilities import SQLDatabase
import os
import json

def get_db_connection():
    """Get a connection to the database from the URI."""
    config = load_config()
    db_uri = config.database.db_uri
    try:
        conn = psycopg2.connect(db_uri)
        print("Connected to database.")
        return conn
    except OperationalError as e:
        print(f"Error connecting to database: {e}")
        raise e
    
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

SCHEMA_CACHE_FILE = "db_schema_cache.json" # Define filename for schema cache
SCHEMA_CACHE_PATH = os.path.join("data", SCHEMA_CACHE_FILE) # Define path for schema cache file

def fetch_database_schema_json() -> str:
    """
    Fetches database schema information as a JSON string using Langchain SQLDatabase.
    """
    db_uri = get_db_connection() # Use existing function to get DB URI
    db = SQLDatabase.from_uri(db_uri) # Initialize SQLDatabase (Langchain Community Package)
    db_schema_string = db.get_table_info() # Get schema info string
    return db_schema_string

def cache_database_schema(schema_json_string: str):
    """
    Caches the database schema information to a JSON file.
    """
    os.makedirs(os.path.dirname(SCHEMA_CACHE_PATH), exist_ok=True) # Ensure directory exists
    try:
        with open(SCHEMA_CACHE_PATH, "w") as f:
            json.dump({"schema_info": schema_json_string}, f, indent=4) # Save as JSON with indent
        print(f"Database schema cached to: {SCHEMA_CACHE_PATH}")
    except Exception as e:
        print(f"Error caching database schema to file: {e}")

def load_database_schema_from_cache() -> str:
    """
    Loads the database schema information from the cache JSON file.
    Returns schema JSON string if loaded successfully, None otherwise.
    """
    if not os.path.exists(SCHEMA_CACHE_PATH):
        print("Schema cache file not found.")
        return None
    try:
        with open(SCHEMA_CACHE_PATH, "r") as f:
            cached_data = json.load(f)
            schema_json_string = cached_data.get("schema_info") # Get schema from JSON
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
    """
    Retrieves database schema information.
    Attempts to load from cache if available and `use_cache` is True, otherwise fetches from DB and caches.
    """
    if use_cache:
        cached_schema = load_database_schema_from_cache() # Try loading from cache
        if cached_schema:
            return cached_schema # Return cached schema if found

    print("Fetching database schema from Neon Postgres...") # Log if fetching from DB
    schema_string = fetch_database_schema_json() # Fetch schema from database
    if schema_string:
        cache_database_schema(schema_string) # Cache the fetched schema for future use
        return schema_string
    else:
        print("Error: Could not fetch database schema from Neon Postgres.")
        return "" # Return empty string if schema fe

def get_vector_db_client():
    """Initialize and returns the ChromaDB client."""
    config = load_config()
    persist_dir = config.chromadb.persist_directory
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        print("ChromaDB client initialized, persisting to directory:", persist_dir)
        return client
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        raise e
    
def get_vector_db_collection(vector_db_client, collection_name="cenomi_mall_data"):
    """Retrieves or creates a collection in the Cenomi mall data."""
    try:
        collection = vector_db_client.get_or_create_collection(name=collection_name)
        print(f"Collection {collection_name} retrieved or created.")
        return collection
    except Exception as e:
        print(f"Error retrieving or creating collection: {e}")
        raise e

def get_vector_db_description(vector_db_collection, limit=5):
    """Retrieves the description of the vector database collection."""
    description = "Vector Database Collection Description: \n\n"
    try:
        sample_data = vector_db_collection.peek(limit=limit)
        description += "Sample Data (limited to {} items):\n".format(limit)
        if sample_data and sample_data['ids']:
            for i in range(len(sample_data['ids'])):
                description += f"\n--- Item {i+1} ---\n"
                if 'ids' in sample_data and sample_data['ids']:
                    description += f"ID: {sample_data['ids'][i]}\n"
                if 'metadatas' in sample_data and sample_data['metadatas'] and sample_data['metadatas'][i]:
                    description += f"Metadata: " + str(sample_data['metadatas'][i])+ "\n"
                if 'documents' in sample_data and sample_data['documents'] and sample_data['documents'][i]:
                    description += f"Document (snippet): " + str(sample_data['documents'][i][:200])+ "...\n"
        else:
            description += "Collection is empty or no sample data could be retrieved.\n"
    except Exception as e:
        description += f"Error retrieving VectorDB description: {e}"
        print("Error retrieving VectorDB description:", e)
    return description

if __name__ == "__main__":
    try:
        neon_conn = get_db_connection()
        schema_desc = get_db_schema_description(neon_conn)
        print("\n--- Database Schema Description ---")
        print(schema_desc)
        neon_conn.close()

        chroma_client = get_vector_db_client()
        chroma_collection = get_vector_db_collection(chroma_client)
        vector_desc = get_vector_db_description(chroma_collection)
        print("\n--- Vector Database Description ---")
        print(vector_desc)

    except Exception as e:
        print(f"An error occurred during testing: {e}")