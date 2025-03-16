import psycopg2
from psycopg2 import OperationalError, errorcodes
from ...config.config_loader import load_config
import chromadb

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