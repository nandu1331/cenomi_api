import time
from agent.utils.database_utils import get_db_connection, get_vector_db_client, get_vector_db_collection
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.config_loader import load_config

config = load_config()

embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
print("HuggingFace embedding model initialized.")

def ingest_data_to_vector_db():
    """
        Ingests data from Neon PostgreSQL database into ChromaDB Vector Database.
    """
    start_time = time.time()
    print("--- Data Ingestion Pipeline Started ---")
    
    try:
        db_conn = get_db_connection()
        vector_db_client = get_vector_db_client()
        vector_db_collection = get_vector_db_collection(vector_db_client)
        
        if not db_conn or not vector_db_client or not vector_db_collection:
            print("Error: Could not access database or VectorDB collection.")
            return

        print("Fetching data from Neon Postgres 'offers' table...")
        offers_data = fetch_offers_data(db_conn)
        if not offers_data:
            print("No data fetched from 'offers' table. Ingestion process stopped for offers.")
            return
        
        print("Processing and ingesting offers data into ChromaDB...")
        ingest_offers_to_chromadb(vector_db_collection, offers_data, embedding_model)
        
        print("Data ingestion pipeline completed successfully.")
    
    except Exception as e:
        print(f"Error during data ingestion pipeline: {e}")
    finally:
        if db_conn:
            db_conn.close() # Ensure database connection is closed
        end_time = time.time()
        duration = end_time - start_time
        print(f"--- Data Ingestion Pipeline Finished in {duration:.2f} seconds ---")
        
def fetch_offers_data(conn):
    """
        Fetches data from the database.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                        SELECT 
                            offer_id,
                            offer_name,
                            offer_description,
                            store_name,
                            product_name,
                            start_date,
                            end_date,
                            discount_percentage
                        FROM offers_view;
                        """)
            offers = cur.fetchall()
            if offers:
                print(f"Successfully fetched {len(offers)} offers from database.")
            else:
                print("No offers found in the database.")
            return offers
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return None
    
def ingest_offers_to_chromadb(chroma_collection, offers_data, embedding_model):
    """
        Ingests offers data into the ChromaDB collection.
    """
    ids = []
    documents = []
    metadatas = []
    
    for offer in offers_data:
        offer_id, offer_name, offer_description, store_name, product_name, start_date, end_date, discount_percentage = offer
        
        document_text = f"""Offer Name: {offer_name}\nOffer Description: {offer_description}\nStore: {store_name}\nProduct: {product_name}\nDiscount: {discount_percentage}%\nStart Date: {start_date}\nEnd Date: {end_date}"""
        documents.append(document_text)
        
        metadata = {
            "source_table": "offers",
            "offer_id": offer_id,
            "store_id": store_name,
            "product_id": product_name,
            "start_date": start_date,
            "end_date": end_date,
            "discount_percentage": discount_percentage
        }
        metadatas.append(metadata)
        ids.append(f"offer_id:{offer_id}")
        
    if documents:
        print(f"Starting to ingest {len(documents)} offers into ChromaDB collection...")
        try:
            chroma_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_model.embed_documents(documents) # Generate embeddings in batch
            )
            print(f"Successfully ingested {len(documents)} offer documents into ChromaDB collection.")
        except Exception as e:
            print(f"Error ingesting offers into ChromaDB: {e}")
    else:
        print("No documents to ingest into ChromaDB.")
        
if __name__ == "__main__":
    ingest_data_to_vector_db()