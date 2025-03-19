import psycopg2
from config.config_loader import load_config
import chromadb
from langchain_community.utilities import SQLDatabase
import os
import json

def get_db_connection():
    config = load_config()
    try:
        return psycopg2.connect(config.database.db_uri)
    except Exception:
        return None

def get_db_schema_description(conn):
    schema_description = "Database Schema:\n\n"
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
    except Exception:
        return ""
    return schema_description

SCHEMA_CACHE_PATH = os.path.join("data", "db_schema_cache.json")

def fetch_database_schema_json() -> str:
    config = load_config()
    db = SQLDatabase.from_uri(config.database.db_uri)
    return db.get_table_info()

def cache_database_schema(schema_json_string: str):
    os.makedirs(os.path.dirname(SCHEMA_CACHE_PATH), exist_ok=True)
    try:
        with open(SCHEMA_CACHE_PATH, "w") as f:
            json.dump({"schema_info": schema_json_string}, f, indent=4)
    except Exception:
        pass

def load_database_schema_from_cache() -> str:
    if not os.path.exists(SCHEMA_CACHE_PATH):
        return None
    try:
        with open(SCHEMA_CACHE_PATH, "r") as f:
            cached_data = json.load(f)
            return cached_data.get("schema_info", None)
    except Exception:
        return None

def get_database_schema_string(use_cache: bool = True) -> str:
    if use_cache:
        cached_schema = load_database_schema_from_cache()
        if cached_schema:
            return cached_schema
    schema_string = fetch_database_schema_json()
    if schema_string:
        cache_database_schema(schema_string)
        return schema_string
    return ""

def get_vector_db_client():
    config = load_config()
    try:
        return chromadb.PersistentClient(path=config.chromadb.persist_directory)
    except Exception:
        return None

def get_vector_db_collection(vector_db_client, collection_name="cenomi_mall_data"):
    try:
        return vector_db_client.get_or_create_collection(name=collection_name)
    except Exception:
        return None

def get_vector_db_description(vector_db_collection, limit=5):
    description = "Vector Database Collection Description:\n\n"
    try:
        sample_data = vector_db_collection.peek(limit=limit)
        description += f"Sample Data (limited to {limit} items):\n"
        if sample_data and sample_data['ids']:
            for i in range(len(sample_data['ids'])):
                description += f"\n--- Item {i+1} ---\n"
                if 'ids' in sample_data and sample_data['ids']:
                    description += f"ID: {sample_data['ids'][i]}\n"
                if 'metadatas' in sample_data and sample_data['metadatas'] and sample_data['metadatas'][i]:
                    description += f"Metadata: {sample_data['metadatas'][i]}\n"
                if 'documents' in sample_data and sample_data['documents'] and sample_data['documents'][i]:
                    description += f"Document (snippet): {sample_data['documents'][i][:200]}...\n"
        else:
            description += "Collection is empty or no sample data could be retrieved.\n"
    except Exception:
        description += "Error retrieving VectorDB description."
    return description