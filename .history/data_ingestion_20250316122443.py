import time
from agent.utils.database_utils import get_db_connection, get_vector_db_client, get_vector_db_collection, fetch_database_schema_json, cache_database_schema
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

        print("Fetching data from Neon Postgres 'offers_view' table...")
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
        print("Fetching data from Neon Postgres 'store_view' table...")
        stores_data = fetch_stores_data(db_conn)
        if stores_data: # Conditionally ingest if data is fetched
            print("Processing and ingesting stores data into ChromaDB...")
            ingest_stores_to_chromadb(vector_db_collection, stores_data, embedding_model)
        else:
            print("Skipping stores ingestion due to no data.")

        # --- Ingest Events Data ---
        print("Fetching data from Neon Postgres 'events_view' table...")
        events_data = fetch_events_data(db_conn)
        if events_data: # Conditionally ingest if data is fetched
            print("Processing and ingesting events data into ChromaDB...")
            ingest_events_to_chromadb(vector_db_collection, events_data, embedding_model)
        else:
            print("Skipping events ingestion due to no data.")
            
        db_schema = fetch_database_schema_json()
        if db_schema:
            cache_database_schema(db_schema)
        else:
            print("Error: Could not fetch database schema from Neon Postgres.")
        
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
        Fetches offers data from the database.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                        SELECT
                            offer_id,
                            offer_name_en,
                            offer_name_ar,
                            offer_description_en,
                            offer_description_ar,
                            store_name_en,
                            store_name_ar,
                            terms_conditions_en,
                            terms_conditions_ar,
                            start_date,
                            end_date,
                            discount_percentage,
                            image_url
                        FROM offers_view
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
        offer_id, offer_name_en, offer_name_ar, offer_description_en, offer_description_ar, store_name_en, store_name_ar, terms_conditions_en, terms_conditions_ar, start_date, end_date, discount_percentage, image_url = offer
        
        document_text = f"""Offer Name (English): {offer_name_en}
Offer Name (Arabic): {offer_name_ar}
Offer Description (English): {offer_description_en}
Offer Description (Arabic): {offer_description_ar}
Store Name (English): {store_name_en}
Store Name (Arabic): {store_name_ar}
Terms & Conditions (English): {terms_conditions_en}
Terms & Conditions (Arabic): {terms_conditions_ar}
Discount: {discount_percentage}%
Valid: {start_date} to {end_date}"""
        documents.append(document_text)
        
        metadata = {
            "source_table": "offers_view",
            "offer_id": offer_id,
            "offer_name_en": offer_name_en,
            "offer_name_ar": offer_name_ar,
            "offer_description_en": offer_description_en,
            "offer_description_ar": offer_description_ar,
            "store_name_en": store_name_en,
            "store_name_ar": store_name_ar,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "discount_percentage": discount_percentage,
            "image_url": image_url
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
                    name_en,
                    name_ar,
                    location_en,
                    location_ar,
                    opening_hours_en,
                    opening_hours_ar,
                    contact_phone,
                    email,
                    geo_latitude,
                    geo_longitude
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
        mall_id, name_en, name_ar, location_en, location_ar, opening_hours_en, opening_hours_ar, contact_phone, email, geo_latitude, geo_longitude = mall

        # Create Document Text Content
        document_text = f"""Mall Name (English): {name_en}
Mall Name (Arabic): {name_ar}
Location (English): {location_en}
Location (Arabic): {location_ar}
Opening Hours (English): {opening_hours_en}
Opening Hours (Arabic): {opening_hours_ar}
Contact Phone: {contact_phone}
Email: {email}"""
        documents.append(document_text)

        # Create Metadata
        metadata = {
            "source_table": "malls",
            "mall_id": mall_id,
            "name_en": name_en,
            "name_ar": name_ar,
            "location_en": location_en,
            "location_ar": location_ar,
            "opening_hours_en": opening_hours_en,
            "opening_hours_ar": opening_hours_ar,
            "contact_phone": contact_phone,
            "email": email,
            "geo_latitude": float(geo_latitude) if geo_latitude is not None else None,
            "geo_longitude": float(geo_longitude) if geo_longitude is not None else None
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
                    store_name_en,
                    store_name_ar,
                    store_description_en,
                    store_description_ar,
                    category_name_en,
                    category_name_ar,
                    location_in_mall_en,
                    location_in_mall_ar,
                    opening_hours_en,
                    opening_hours_ar,
                    contact_phone,
                    email,
                    mall_name_en,
                    mall_name_ar,
                    logo_url
                FROM store_view;
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
        store_id, store_name_en, store_name_ar, store_description_en, store_description_ar, category_name_en, category_name_ar, location_in_mall_en, location_in_mall_ar, opening_hours_en, opening_hours_ar, contact_phone, email, mall_name_en, mall_name_ar, logo_url = store

        # Create Document Text Content
        document_text = f"""Store Name (English): {store_name_en}
Store Name (Arabic): {store_name_ar}
Description (English): {store_description_en}
Description (Arabic): {store_description_ar}
Category (English): {category_name_en}
Category (Arabic): {category_name_ar}
Mall (English): {mall_name_en}
Mall (Arabic): {mall_name_ar}
Location in Mall (English): {location_in_mall_en}
Location in Mall (Arabic): {location_in_mall_ar}
Opening Hours (English): {opening_hours_en}
Opening Hours (Arabic): {opening_hours_ar}
Contact Phone: {contact_phone}
Email: {email}"""
        documents.append(document_text)

        # Create Metadata
        metadata = {
            "source_table": "store_view",
            "store_id": store_id,
            "store_name_en": store_name_en,
            "store_name_ar": store_name_ar,
            "store_description_en": store_description_en,
            "store_description_ar": store_description_ar,
            "category_name_en": category_name_en,
            "category_name_ar": category_name_ar,
            "location_in_mall_en": location_in_mall_en,
            "location_in_mall_ar": location_in_mall_ar,
            "mall_name_en": mall_name_en, 
            "mall_name_ar": mall_name_ar,
            "opening_hours_en": opening_hours_en,
            "opening_hours_ar": opening_hours_ar,
            "contact_phone": contact_phone,
            "email": email,
            "logo_url": logo_url
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
                    event_name_en,
                    event_name_ar,
                    event_description_en,
                    event_description_ar,
                    start_date,
                    end_date,
                    location_in_mall_en,
                    location_in_mall_ar,
                    mall_name_en,
                    mall_name_ar,
                    image_url
                FROM events_view;
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
    Ingests events data into the ChromaDB collection.
    """
    ids = []
    documents = []
    metadatas = []

    for event in events_data:
        (event_id, event_name_en, event_name_ar, event_description_en, event_description_ar,
         start_date, end_date, location_in_mall_en, location_in_mall_ar,
         mall_name_en, mall_name_ar, image_url) = event

        # Create Document Text Content
        document_text = f"""Event Name (English): {event_name_en}
Event Name (Arabic): {event_name_ar}
Event Description (English): {event_description_en}
Event Description (Arabic): {event_description_ar}
Start Date: {start_date}
End Date: {end_date}
Location in Mall (English): {location_in_mall_en}
Location in Mall (Arabic): {location_in_mall_ar}
Mall Name (English): {mall_name_en}
Mall Name (Arabic): {mall_name_ar}
Image URL: {image_url}"""
        documents.append(document_text)

        # Create Metadata for the event
        metadata = {
            "source_table": "events_view",
            "event_id": event_id,
            "event_name_en": event_name_en,
            "event_name_ar": event_name_ar,
            "event_description_en": event_description_en,
            "event_description_ar": event_description_ar,
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "location_in_mall_en": location_in_mall_en,
            "location_in_mall_ar": location_in_mall_ar,
            "mall_name_en": mall_name_en,
            "mall_name_ar": mall_name_ar,
            "image_url": image_url
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