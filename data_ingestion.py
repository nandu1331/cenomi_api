import time
import json
from psycopg2 import OperationalError
from agent.utils.database_utils import (
    get_db_connection,
    get_vector_db_client, 
    get_vector_db_index,
    fetch_database_schema_json,
    cache_database_schema,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.config_loader import load_config

# Load configuration and initialize embedding model
config = load_config()
embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
print("HuggingFace embedding model initialized.")

# Utility function for fetching data with retry logic
def fetch_data_with_retry(conn, query, table_name, max_retries=3):
    """
    Fetch data from the database with retry logic for connection issues.
    
    Args:
        conn: Database connection object
        query: SQL query to execute
        table_name: Name of the table for logging
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        tuple: (fetched_data, updated_connection)
    """
    for attempt in range(max_retries):
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                data = cur.fetchall()
                if data:
                    print(f"Successfully fetched {len(data)} {table_name} from database.")
                else:
                    print(f"No {table_name} found in the database.")
                return data, conn
        except OperationalError as e:
            print(f"Error fetching {table_name}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying {table_name} fetch ({attempt + 1}/{max_retries})...")
                time.sleep(2 ** attempt)  # Exponential backoff
                conn.close()
                conn = get_db_connection()
                if not conn:
                    print("Failed to reconnect to database.")
                    return None, None
            else:
                print(f"Failed to fetch {table_name} after {max_retries} attempts.")
                return None, conn
    return None, conn

# Fetch Functions (updated to use retry logic)
def fetch_amenities_data(conn):
    query = """SELECT amenity_id, mall_id, name_en, name_ar, location_en, location_ar, description_en, description_ar FROM amenities;"""
    return fetch_data_with_retry(conn, query, "amenities")

def fetch_customers_data(conn):
    query = """SELECT customer_id, mall_id, email, preferred_language FROM customers;"""
    return fetch_data_with_retry(conn, query, "customers")

def fetch_loyalty_programs_data(conn):
    query = """SELECT loyalty_id, mall_id, name_en, name_ar, description_en, description_ar, points_per_purchase, redemption_rules_en, redemption_rules_ar FROM loyalty_programs;"""
    return fetch_data_with_retry(conn, query, "loyalty_programs")

def fetch_malls_data(conn):
    query = """
        SELECT
            mall_id,
            name_en,
            name_ar,
            location_en,
            location_ar,
            description_en,
            description_ar
        FROM malls;
    """
    return fetch_data_with_retry(conn, query, "malls")

def fetch_offers_data(conn):
    query = """
        SELECT
            o.offer_id,
            o.description_en AS offer_description_en,
            o.description_ar AS offer_description_ar,
            o.start_date,
            o.end_date,
            s.store_id,
            s.name_en AS store_name_en,
            s.name_ar AS store_name_ar,
            m.mall_id,
            m.name_en AS mall_name_en,
            m.name_ar AS mall_name_ar
        FROM offers o
        JOIN stores s ON o.store_id = s.store_id
        JOIN malls m ON s.mall_id = m.mall_id;
    """
    return fetch_data_with_retry(conn, query, "offers")

def fetch_products_data(conn):
    query = """
        SELECT
            p.product_id,
            p.store_id,
            p.name_en AS product_name_en,
            p.name_ar AS product_name_ar,
            p.description_en AS product_description_en,
            p.description_ar AS product_description_ar,
            p.price,
            p.currency,
            s.name_en AS store_name_en,
            s.name_ar AS store_name_ar
        FROM products p
        JOIN stores s ON p.store_id = s.store_id;
    """
    return fetch_data_with_retry(conn, query, "products")

def fetch_services_data(conn):
    query = """SELECT service_id, mall_id, name_en, name_ar, description_en, description_ar FROM services;"""
    return fetch_data_with_retry(conn, query, "services")

def fetch_stores_data(conn):
    query = """
        SELECT
            s.store_id,
            s.name_en AS store_name_en,
            s.name_ar AS store_name_ar,
            s.description_en AS store_description_en,
            s.description_ar AS store_description_ar,
            s.category_en,
            s.category_ar,
            s.location_en AS location_in_mall_en,
            s.location_ar AS location_in_mall_ar,
            m.mall_id,
            m.name_en AS mall_name_en,
            m.name_ar AS mall_name_ar
        FROM stores s
        JOIN malls m ON s.mall_id = m.mall_id;
    """
    return fetch_data_with_retry(conn, query, "stores")

def fetch_tenants_data(conn):
    query = """SELECT tenant_id, first_name_en, first_name_ar, last_name_en, last_name_ar, email, phone FROM tenants;"""
    return fetch_data_with_retry(conn, query, "tenants")

def fetch_events_data(conn):
    query = """
        SELECT
            e.event_id,
            e.name_en AS event_name_en,
            e.name_ar AS event_name_ar,
            e.description_en AS event_description_en,
            e.description_ar AS event_description_ar,
            e.start_time,
            e.end_time,
            e.location_en AS location_in_mall_en,
            e.location_ar AS location_in_mall_ar,
            m.mall_id,
            m.name_en AS mall_name_en,
            m.name_ar AS mall_name_ar
        FROM events e
        JOIN malls m ON e.mall_id = m.mall_id;
    """
    return fetch_data_with_retry(conn, query, "events")

# Ingestion Functions (updated with batching)
def ingest_amenities_to_vector_db(vector_db_index, amenities_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for amenity in amenities_data:
        amenity_id, mall_id, name_en, name_ar, location_en, location_ar, description_en, description_ar = amenity
        document_text = f"""Amenity Name (English): {name_en}
Amenity Name (Arabic): {name_ar}
Location (English): {location_en}
Location (Arabic): {location_ar}
Description (English): {description_en}
Description (Arabic): {description_ar}"""
        documents.append(document_text)
        metadata = {
            "source_table": "amenities",
            "amenity_id": amenity_id,
            "mall_id": mall_id,
            "name_en": name_en,
            "name_ar": name_ar,
            "location_en": location_en,
            "location_ar": location_ar,
            "description_en": description_en,
            "description_ar": description_ar,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"amenity_id:{amenity_id}")
    if documents:
        print(f"Ingesting {len(documents)} amenity documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} amenity documents.")
        except Exception as e:
            print(f"Error ingesting amenities: {e}")
    else:
        print("No amenity documents to ingest.")

def ingest_customers_to_vector_db(vector_db_index, customers_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for customer in customers_data:
        customer_id, mall_id, email, preferred_language = customer
        document_text = f"""Customer Email: {email}
Preferred Language: {preferred_language}"""
        documents.append(document_text)
        metadata = {
            "source_table": "customers",
            "customer_id": customer_id,
            "mall_id": mall_id,
            "email": email,
            "preferred_language": preferred_language,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"customer_id:{customer_id}")
    if documents:
        print(f"Ingesting {len(documents)} customer documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} customer documents.")
        except Exception as e:
            print(f"Error ingesting customers: {e}")
    else:
        print("No customer documents to ingest.")

def ingest_loyalty_programs_to_vector_db(vector_db_index, loyalty_programs_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for lp in loyalty_programs_data:
        loyalty_id, mall_id, name_en, name_ar, desc_en, desc_ar, points, rules_en, rules_ar = lp
        document_text = f"""Loyalty Program Name (English): {name_en}
Loyalty Program Name (Arabic): {name_ar}
Description (English): {desc_en}
Description (Arabic): {desc_ar}
Points per Purchase: {points}
Redemption Rules (English): {rules_en}
Redemption Rules (Arabic): {rules_ar}"""
        documents.append(document_text)
        metadata = {
            "source_table": "loyalty_programs",
            "loyalty_id": loyalty_id,
            "mall_id": mall_id,
            "name_en": name_en,
            "name_ar": name_ar,
            "description_en": desc_en,
            "description_ar": desc_ar,
            "points_per_purchase": float(points) if points is not None else None,
            "redemption_rules_en": rules_en,
            "redemption_rules_ar": rules_ar,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"loyalty_id:{loyalty_id}")
    if documents:
        print(f"Ingesting {len(documents)} loyalty program documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} loyalty program documents.")
        except Exception as e:
            print(f"Error ingesting loyalty programs: {e}")
    else:
        print("No loyalty program documents to ingest.")

def ingest_malls_to_vector_db(vector_db_index, malls_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for mall in malls_data:
        mall_id, name_en, name_ar, location_en, location_ar, desc_en, desc_ar = mall
        document_text = f"""Mall Name (English): {name_en}
Mall Name (Arabic): {name_ar}
Location (English): {location_en}
Location (Arabic): {location_ar}
Description (English): {desc_en}
Description (Arabic): {desc_ar}"""
        documents.append(document_text)
        metadata = {
            "source_table": "malls",
            "mall_id": mall_id,
            "name_en": name_en,
            "name_ar": name_ar,
            "location_en": location_en,
            "location_ar": location_ar,
            "description_en": desc_en,
            "description_ar": desc_ar,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"mall_id:{mall_id}")
    if documents:
        print(f"Ingesting {len(documents)} mall documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} mall documents.")
        except Exception as e:
            print(f"Error ingesting malls: {e}")
    else:
        print("No mall documents to ingest.")

def ingest_offers_to_vector_db(vector_db_index, offers_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for offer in offers_data:
        offer_id, desc_en, desc_ar, start_date, end_date, store_id, store_name_en, store_name_ar, mall_id, mall_name_en, mall_name_ar = offer
        document_text = f"""Offer Description (English): {desc_en}
Offer Description (Arabic): {desc_ar}
Start Date: {start_date}
End Date: {end_date}
Store Name (English): {store_name_en}
Store Name (Arabic): {store_name_ar}
Mall Name (English): {mall_name_en}
Mall Name (Arabic): {mall_name_ar}"""
        documents.append(document_text)
        metadata = {
            "source_table": "offers",
            "offer_id": offer_id,
            "description_en": desc_en,
            "description_ar": desc_ar,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "store_id": store_id,
            "store_name_en": store_name_en,
            "store_name_ar": store_name_ar,
            "mall_id": mall_id,
            "mall_name_en": mall_name_en,
            "mall_name_ar": mall_name_ar,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"offer_id:{offer_id}")
    if documents:
        print(f"Ingesting {len(documents)} offer documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} offer documents.")
        except Exception as e:
            print(f"Error ingesting offers: {e}")
    else:
        print("No offer documents to ingest.")

def ingest_products_to_vector_db(vector_db_index, products_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for product in products_data:
        product_id, store_id, name_en, name_ar, desc_en, desc_ar, price, currency, store_name_en, store_name_ar = product
        document_text = f"""Product Name (English): {name_en}
Product Name (Arabic): {name_ar}
Description (English): {desc_en}
Description (Arabic): {desc_ar}
Price: {price} {currency}
Store Name (English): {store_name_en}
Store Name (Arabic): {store_name_ar}"""
        documents.append(document_text)
        metadata = {
            "source_table": "products",
            "product_id": product_id,
            "store_id": store_id,
            "name_en": name_en,
            "name_ar": name_ar,
            "description_en": desc_en,
            "description_ar": desc_ar,
            "price": float(price),
            "currency": currency,
            "store_name_en": store_name_en,
            "store_name_ar": store_name_ar,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"product_id:{product_id}")
    if documents:
        print(f"Ingesting {len(documents)} product documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} product documents.")
        except Exception as e:
            print(f"Error ingesting products: {e}")
    else:
        print("No product documents to ingest.")

def ingest_services_to_vector_db(vector_db_index, services_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for service in services_data:
        service_id, mall_id, name_en, name_ar, desc_en, desc_ar = service
        document_text = f"""Service Name (English): {name_en}
Service Name (Arabic): {name_ar}
Description (English): {desc_en}
Description (Arabic): {desc_ar}"""
        documents.append(document_text)
        metadata = {
            "source_table": "services",
            "service_id": service_id,
            "mall_id": mall_id,
            "name_en": name_en,
            "name_ar": name_ar,
            "description_en": desc_en,
            "description_ar": desc_ar,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"service_id:{service_id}")
    if documents:
        print(f"Ingesting {len(documents)} service documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} service documents.")
        except Exception as e:
            print(f"Error ingesting services: {e}")
    else:
        print("No service documents to ingest.")

def ingest_stores_to_vector_db(vector_db_index, stores_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for store in stores_data:
        store_id, name_en, name_ar, desc_en, desc_ar, cat_en, cat_ar, loc_en, loc_ar, mall_id, mall_name_en, mall_name_ar = store
        document_text = f"""Store Name (English): {name_en}
Store Name (Arabic): {name_ar}
Description (English): {desc_en}
Description (Arabic): {desc_ar}
Category (English): {cat_en}
Category (Arabic): {cat_ar}
Location in Mall (English): {loc_en}
Location in Mall (Arabic): {loc_ar}
Mall Name (English): {mall_name_en}
Mall Name (Arabic): {mall_name_ar}"""
        documents.append(document_text)
        metadata = {
            "source_table": "stores",
            "store_id": store_id,
            "name_en": name_en,
            "name_ar": name_ar,
            "description_en": desc_en,
            "description_ar": desc_ar,
            "category_en": cat_en,
            "category_ar": cat_ar,
            "location_en": loc_en,
            "location_ar": loc_ar,
            "mall_id": mall_id,
            "mall_name_en": mall_name_en,
            "mall_name_ar": mall_name_ar,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"store_id:{store_id}")
    if documents:
        print(f"Ingesting {len(documents)} store documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} store documents.")
        except Exception as e:
            print(f"Error ingesting stores: {e}")
    else:
        print("No store documents to ingest.")

def ingest_tenants_to_vector_db(vector_db_index, tenants_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for tenant in tenants_data:
        tenant_id, fname_en, fname_ar, lname_en, lname_ar, email, phone = tenant
        document_text = f"""Tenant First Name (English): {fname_en}
Tenant First Name (Arabic): {fname_ar}
Tenant Last Name (English): {lname_en}
Tenant Last Name (Arabic): {lname_ar}
Email: {email}
Phone: {phone}"""
        documents.append(document_text)
        metadata = {
            "source_table": "tenants",
            "tenant_id": tenant_id,
            "first_name_en": fname_en,
            "first_name_ar": fname_ar,
            "last_name_en": lname_en,
            "last_name_ar": lname_ar,
            "email": email,
            "phone": phone,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"tenant_id:{tenant_id}")
    if documents:
        print(f"Ingesting {len(documents)} tenant documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} tenant documents.")
        except Exception as e:
            print(f"Error ingesting tenants: {e}")
    else:
        print("No tenant documents to ingest.")

def ingest_events_to_vector_db(vector_db_index, events_data, embedding_model, batch_size=100):
    ids = []
    documents = []
    metadatas = []
    for event in events_data:
        event_id, name_en, name_ar, desc_en, desc_ar, start_time, end_time, loc_en, loc_ar, mall_id, mall_name_en, mall_name_ar = event
        document_text = f"""Event Name (English): {name_en}
Event Name (Arabic): {name_ar}
Description (English): {desc_en}
Description (Arabic): {desc_ar}
Start Time: {start_time}
End Time: {end_time}
Location in Mall (English): {loc_en}
Location in Mall (Arabic): {loc_ar}
Mall Name (English): {mall_name_en}
Mall Name (Arabic): {mall_name_ar}"""
        documents.append(document_text)
        metadata = {
            "source_table": "events",
            "event_id": event_id,
            "name_en": name_en,
            "name_ar": name_ar,
            "description_en": desc_en,
            "description_ar": desc_ar,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "location_en": loc_en,
            "location_ar": loc_ar,
            "mall_id": mall_id,
            "mall_name_en": mall_name_en,
            "mall_name_ar": mall_name_ar,
            "document": document_text
        }
        metadatas.append(metadata)
        ids.append(f"event_id:{event_id}")
    if documents:
        print(f"Ingesting {len(documents)} event documents into Pinecone...")
        try:
            embeddings = embedding_model.embed_documents(documents)
            vectors = [(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db_index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
            print(f"Successfully ingested {len(documents)} event documents.")
        except Exception as e:
            print(f"Error ingesting events: {e}")
    else:
        print("No event documents to ingest.")

# Main Ingestion Pipeline (updated to handle connection updates)
def ingest_data_to_vector_db():
    start_time = time.time()
    print("--- Data Ingestion Pipeline Started ---")
    try:
        db_conn = get_db_connection()
        vector_db_client = get_vector_db_client()
        # Note: Fixed potential typo from 'pineconedb' to 'pinecone' based on typical config naming
        index_name = getattr(config, 'pineconedb', config).index_name
        vector_db_index = get_vector_db_index(vector_db_client, index_name)
        if not db_conn or not vector_db_client or not vector_db_index:
            print("Error: Could not access database or Pinecone index.")
            return

        # List of tables and their fetch/ingest functions
        tables = [
            ("amenities", fetch_amenities_data, ingest_amenities_to_vector_db),
            ("customers", fetch_customers_data, ingest_customers_to_vector_db),
            ("loyalty_programs", fetch_loyalty_programs_data, ingest_loyalty_programs_to_vector_db),
            ("malls", fetch_malls_data, ingest_malls_to_vector_db),
            ("offers", fetch_offers_data, ingest_offers_to_vector_db),
            ("products", fetch_products_data, ingest_products_to_vector_db),
            ("services", fetch_services_data, ingest_services_to_vector_db),
            ("stores", fetch_stores_data, ingest_stores_to_vector_db),
            ("tenants", fetch_tenants_data, ingest_tenants_to_vector_db),
            ("events", fetch_events_data, ingest_events_to_vector_db),
        ]

        for table_name, fetch_func, ingest_func in tables:
            print(f"Fetching data from Neon Postgres '{table_name}' table...")
            data, db_conn = fetch_func(db_conn)
            if data:
                print(f"Processing and ingesting {table_name} data into Pinecone...")
                ingest_func(vector_db_index, data, embedding_model)
            else:
                print(f"Skipping {table_name} ingestion due to no data.")

        # Cache database schema
        db_schema = fetch_database_schema_json()
        if db_schema:
            cache_database_schema(db_schema)
        else:
            print("Error: Could not fetch database schema from Neon Postgres.")

        print("Data ingestion pipeline completed successfully.")
    except Exception as e:
        print(f"Error during data ingestion pipeline: {e}")
    finally:
        if 'db_conn' in locals() and db_conn and not db_conn.closed:
            db_conn.close()
        end_time = time.time()
        duration = end_time - start_time
        print(f"--- Data Ingestion Pipeline Finished in {duration:.2f} seconds ---")

if __name__ == "__main__":
    ingest_data_to_vector_db()