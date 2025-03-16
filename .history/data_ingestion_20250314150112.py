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
        if offers_data: # Conditionally ingest if data is fetched
            print("Processing and ingesting offers data into ChromaDB...")
            ingest_offers_to_chromadb(vector_db_collection, offers_data, embedding_model)
        else:
            print("Skipping offers ingestion due to no data.")

        # --- Ingest Malls Data ---
        print("Fetching data from Neon Postgres 'malls' table...")
        malls_data = fetch_malls_data(db_conn)
        if malls_data: # Conditionally ingest if data is fetched
            print("Processing and ingesting malls data into ChromaDB...")
            ingest_malls_to_chromadb(vector_db_collection, malls_data, embedding_model)
        else:
            print("Skipping malls ingestion due to no data.")

        # --- Ingest Stores Data ---
        print("Fetching data from Neon Postgres 'stores' table...")
        stores_data = fetch_stores_data(db_conn)
        if stores_data: # Conditionally ingest if data is fetched
            print("Processing and ingesting stores data into ChromaDB...")
            ingest_stores_to_chromadb(vector_db_collection, stores_data, embedding_model)
        else:
            print("Skipping stores ingestion due to no data.")

        # --- Ingest Events Data ---
        print("Fetching data from Neon Postgres 'events' table...")
        events_data = fetch_events_data(db_conn)
        if events_data: # Conditionally ingest if data is fetched
            print("Processing and ingesting events data into ChromaDB...")
            ingest_events_to_chromadb(vector_db_collection, events_data, embedding_model)
        else:
            print("Skipping events ingestion due to no data.")
        
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
    
def ingest_offers_to_chromadb(vector_db_collection, offers_data, embedding_model):
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
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "discount_percentage": discount_percentage
        }
        metadatas.append(metadata)
        ids.append(f"offer_id:{offer_id}")
        
    if documents:
        print(f"Starting to ingest {len(documents)} offers into ChromaDB collection...")
        try:
            vector_db_collection.add(
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
        
def fetch_malls_data(conn):
    """
    Fetches mall data from the Neon PostgreSQL database.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    mall_id,
                    name,
                    location,
                    operating_hours
                FROM malls;
            """)
            malls = cur.fetchall()
            if malls:
                print(f"Successfully fetched {len(malls)} malls from database.")
            else:
                print("No malls found in the database.")
            return malls
    except Exception as e:
        print(f"Error fetching malls data from database: {e}")
        return None
    
def ingest_malls_to_chromadb(vector_db_collection, malls_data, embedding_model):
    """
    Ingests mall data into the ChromaDB collection.
    """
    ids = []
    documents = []
    metadatas = []

    for mall in malls_data:
        mall_id, name, location, operating_hours = mall

        # 1. Create Document Text Content
        document_text = f"""Mall Name: {name}\nLocation: {location}\nOperating Hours: {operating_hours}"""
        documents.append(document_text)

        # 2. Create Metadata
        metadata = {
            "source_table": "malls",
            "mall_id": mall_id,
            "mall_name": name,
            "location": location,
            "operating_hours": operating_hours,
        }
        metadatas.append(metadata)
        ids.append(f"mall_id:{mall_id}")

    if documents:
        print(f"Generating embeddings and adding {len(documents)} mall documents to ChromaDB...")
        try:
            vector_db_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_model.embed_documents(documents)
            )
            print(f"Successfully ingested {len(documents)} mall documents into ChromaDB collection.")
        except Exception as e:
            print(f"Error adding mall documents to ChromaDB: {e}")
    else:
        print("No mall documents to ingest into ChromaDB.")
        
def fetch_stores_data(conn):
    """
    Fetches store data from the Neon PostgreSQL database.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    store_id,
                    store_name,
                    category,
                    location_in_mall,
                    operating_hours,
                    contact_phone,
                    mall_name, -- Assuming you can join to get mall_name
                    tenant_name -- Assuming you can join to get tenant_name
                FROM stores_view; -- Assuming you have a view 'stores_view'
            """)
            stores = cur.fetchall()
            if stores:
                print(f"Successfully fetched {len(stores)} stores from database.")
            else:
                print("No stores found in the database.")
            return stores
    except Exception as e:
        print(f"Error fetching stores data from database: {e}")
        return None
    
def ingest_stores_to_chromadb(vector_db_collection, stores_data, embedding_model):
    """
    Ingests store data into the ChromaDB collection.
    """
    ids = []
    documents = []
    metadatas = []

    for store in stores_data:
        store_id, name, category, location_in_mall, operating_hours, contact_phone, mall_name, tenant_name = store

        # 1. Create Document Text Content
        document_text = f"""Store Name: {name}\nCategory: {category}\nMall: {mall_name}\nTenant: {tenant_name}\nLocation in Mall: {location_in_mall}\nOperating Hours: {operating_hours}\nContact Phone: {contact_phone}"""
        documents.append(document_text)

        # 2. Create Metadata
        metadata = {
            "source_table": "stores",
            "store_id": store_id,
            "store_name": name,
            "category": category,
            "mall_name": mall_name,
            "tenant_name": tenant_name,
            "location_in_mall": location_in_mall,
            "operating_hours": operating_hours,
            "contact_phone": contact_phone,
        }
        metadatas.append(metadata)
        ids.append(f"store_id:{store_id}")

    if documents:
        print(f"Generating embeddings and adding {len(documents)} store documents to ChromaDB...")
        try:
            vector_db_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_model.embed_documents(documents)
            )
            print(f"Successfully ingested {len(documents)} store documents into ChromaDB collection.")
        except Exception as e:
            print(f"Error adding store documents to ChromaDB: {e}")
    else:
        print("No store documents to ingest into ChromaDB.")
        
def fetch_events_data(conn):
    """
    Fetches event data from the Neon PostgreSQL database.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    event_id,
                    title,
                    description,
                    event_date,
                    start_time,
                    end_time,
                    mall_name -- Assuming you can join to get mall_name
                FROM events_view; -- Assuming you have a view 'events_view'
            """)
            events = cur.fetchall()
            if events:
                print(f"Successfully fetched {len(events)} events from database.")
            else:
                print("No events found in the database.")
            return events
    except Exception as e:
        print(f"Error fetching events data from database: {e}")
        return None
    
def ingest_events_to_chromadb(vector_db_collection, events_data, embedding_model):
    """
    Ingests event data into the ChromaDB collection.
    """
    ids = []
    documents = []
    metadatas = []

    for event in events_data:
        event_id, title, description, event_date, start_time, end_time, mall_name = event

        # 1. Create Document Text Content
        document_text = f"""Event Title: {title}\nDescription: {description}\nMall: {mall_name}\nDate: {event_date}\nStart Time: {start_time}\nEnd Time: {end_time}"""
        documents.append(document_text)

        # 2. Create Metadata
        metadata = {
            "source_table": "events",
            "event_id": event_id,
            "event_title": title,
            "mall_name": mall_name,
            "event_date": str(event_date) if event_date else None,
            "start_time": str(start_time) if start_time else None,
            "end_time": str(end_time) if end_time else None,
        }
        metadatas.append(metadata)
        ids.append(f"event_id:{event_id}")

    if documents:
        print(f"Generating embeddings and adding {len(documents)} event documents to ChromaDB...")
        try:
            vector_db_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_model.embed_documents(documents)
            )
            print(f"Successfully ingested {len(documents)} event documents into ChromaDB collection.")
        except Exception as e:
            print(f"Error adding event documents to ChromaDB: {e}")
    else:
        print("No event documents to ingest into ChromaDB.")
        
if __name__ == "__main__":
    ingest_data_to_vector_db()