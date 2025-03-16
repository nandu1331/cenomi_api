# from nodes.intent_router_node import IntentCategory
from typing import List, Dict, Any
# from utils.database_utils import load_database_schema_from_cache
import re

class MockIntentCategory:
        TENANT_UPDATE_OFFER = "TENANT_UPDATE_OFFER"
        TENANT_INSERT_OFFER = "TENANT_INSERT_OFFER"
        TENANT_DELETE_OFFER = "TENANT_DELETE_OFFER"
        TENANT_UPDATE_STORE = "TENANT_UPDATE_STORE"
        TENANT_INSERT_STORE = "TENANT_INSERT_STORE"

def parse_sql_schema(schema_sql: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse SQL CREATE TABLE statements into a structured dictionary.
    
    Args:
        schema_sql (str): SQL schema as CREATE TABLE statements
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping table names to their column information
    """
    tables = {}
    
    # Extract CREATE TABLE blocks
    table_blocks = re.findall(r'CREATE TABLE (\w+) \(([\s\S]*?)\)(?:;|\n\n)', schema_sql)
    
    for table_name, columns_text in table_blocks:
        tables[table_name.lower()] = {"columns": {}}
        
        # Extract column definitions
        column_defs = columns_text.strip().split(',\n\t')
        
        for column_def in column_defs:
            column_def = column_def.strip()
            
            # Skip constraints
            if column_def.startswith('CONSTRAINT'):
                continue
                
            # Extract column name and properties
            match = re.match(r'(\w+)\s+([\w\s\(\)]+)(?:\s+(.*))?', column_def)
            if match:
                col_name, col_type, constraints = match.groups()
                constraints = constraints or ""
                
                # Create column info dictionary
                column_info = {
                    "name": col_name,
                    "type": col_type.strip(),
                    "nullable": "NOT NULL" not in constraints,
                    "primary_key": "PRIMARY KEY" in constraints,
                    "auto_increment": "SERIAL" in col_type.upper() or "IDENTITY" in constraints.upper()
                }
                
                tables[table_name.lower()]["columns"][col_name] = column_info
                
    return tables

def calculate_required_fields(intent: MockIntentCategory, entity_type: str) -> List[str]:
    """
    (Function ii)
    Calculates the list of required fields for a tenant data operation based on intent and entity type.
    Uses cached schema information to determine required fields dynamically.
    
    Args:
        intent (MockIntentCategory): The tenant intent (e.g., TENANT_INSERT_OFFER).
        entity_type (str): The entity type (e.g., "offer", "store").
    
    Returns:
        List[str]: A list of required field names for the operation.
                  Returns an empty list if no required fields are determined or in case of errors.
    """
    print(f"calculate_required_fields - Intent: {intent}, Entity Type: {entity_type}")
    operation_type = None
    
    # Determine operation type from intent
    if intent in [MockIntentCategory.TENANT_UPDATE_OFFER, MockIntentCategory.TENANT_UPDATE_STORE]:
        operation_type = "update"
    elif intent in [MockIntentCategory.TENANT_INSERT_OFFER, MockIntentCategory.TENANT_INSERT_STORE]:
        operation_type = "insert"
    elif intent == MockIntentCategory.TENANT_DELETE_OFFER:
        operation_type = "delete"
    else:
        print(f"calculate_required_fields - Unknown intent for tenant data operation: {intent}. Returning empty required fields.")
        return [] # Unknown intent, return no required fields
    
    try:
        # Load raw schema from cache
        raw_schema = "\nCREATE TABLE chat_history (\n\tchat_id SERIAL NOT NULL, \n\tcustomer_id INTEGER, \n\ttenant_id INTEGER, \n\tis_customer BOOLEAN NOT NULL, \n\tmessage TEXT NOT NULL, \n\tresponse TEXT, \n\tquery_type VARCHAR(50), \n\ttimestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP, \n\tsentiment_score NUMERIC(3, 2), \n\tresolved BOOLEAN DEFAULT false, \n\tCONSTRAINT chat_history_pkey PRIMARY KEY (chat_id), \n\tCONSTRAINT chat_history_customer_id_fkey FOREIGN KEY(customer_id) REFERENCES customers (customer_id) ON DELETE SET NULL, \n\tCONSTRAINT chat_history_tenant_id_fkey FOREIGN KEY(tenant_id) REFERENCES tenant_users (user_id) ON DELETE SET NULL\n)\n\n/*\n3 rows from chat_history table:\nchat_id\tcustomer_id\ttenant_id\tis_customer\tmessage\tresponse\tquery_type\ttimestamp\tsentiment_score\tresolved\n\n*/\n\n\nCREATE TABLE customers (\n\tcustomer_id SERIAL NOT NULL, \n\tfull_name VARCHAR(100) NOT NULL, \n\temail VARCHAR(100) NOT NULL, \n\tphone VARCHAR(20) NOT NULL, \n\tloyalty_points INTEGER DEFAULT 0, \n\tregistration_date DATE NOT NULL, \n\tpreferred_language VARCHAR(10) DEFAULT 'en'::character varying, \n\tCONSTRAINT customers_pkey PRIMARY KEY (customer_id), \n\tCONSTRAINT customers_email_key UNIQUE (email), \n\tCONSTRAINT customers_phone_key UNIQUE (phone)\n)\n\n/*\n3 rows from customers table:\ncustomer_id\tfull_name\temail\tphone\tloyalty_points\tregistration_date\tpreferred_language\n\n*/\n\n\nCREATE TABLE events (\n\tevent_id SERIAL NOT NULL, \n\tmall_id INTEGER, \n\tname_en VARCHAR(100) NOT NULL, \n\tname_ar VARCHAR(100) NOT NULL, \n\tdescription_en TEXT NOT NULL, \n\tdescription_ar TEXT NOT NULL, \n\tstart_date DATE NOT NULL, \n\tend_date DATE NOT NULL, \n\tlocation_in_mall_en VARCHAR(255) NOT NULL, \n\tlocation_in_mall_ar VARCHAR(255) NOT NULL, \n\timage_url VARCHAR(255), \n\tCONSTRAINT events_pkey PRIMARY KEY (event_id), \n\tCONSTRAINT events_mall_id_fkey FOREIGN KEY(mall_id) REFERENCES malls (mall_id) ON DELETE CASCADE\n)\n\n/*\n3 rows from events table:\nevent_id\tmall_id\tname_en\tname_ar\tdescription_en\tdescription_ar\tstart_date\tend_date\tlocation_in_mall_en\tlocation_in_mall_ar\timage_url\n1\t1\tSpring Fashion Festival\t\u0645\u0647\u0631\u062c\u0627\u0646 \u0623\u0632\u064a\u0627\u0621 \u0627\u0644\u0631\u0628\u064a\u0639\tDiscover the latest spring collections from top brands.\t\u0627\u0643\u062a\u0634\u0641 \u0623\u062d\u062f\u062b \u0645\u062c\u0645\u0648\u0639\u0627\u062a \u0627\u0644\u0631\u0628\u064a\u0639 \u0645\u0646 \u0623\u0641\u0636\u0644 \u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a \u0627\u0644\u062a\u062c\u0627\u0631\u064a\u0629.\t2025-03-20\t2025-03-25\tCentral Atrium\t\u0627\u0644\u0631\u062f\u0647\u0629 \u0627\u0644\u0645\u0631\u0643\u0632\u064a\u0629\thttps://cenomi.com/events/spring_fashion.jpg\n2\t2\tSeafood Week\t\u0623\u0633\u0628\u0648\u0639 \u0627\u0644\u0645\u0623\u0643\u0648\u0644\u0627\u062a \u0627\u0644\u0628\u062d\u0631\u064a\u0629\tEnjoy seafood dishes from various cuisines.\t\u0627\u0633\u062a\u0645\u062a\u0639 \u0628\u0623\u0637\u0628\u0627\u0642 \u0627\u0644\u0645\u0623\u0643\u0648\u0644\u0627\u062a \u0627\u0644\u0628\u062d\u0631\u064a\u0629 \u0645\u0646 \u0645\u062e\u062a\u0644\u0641 \u0627\u0644\u0645\u0637\u0627\u0628\u062e.\t2025-04-01\t2025-04-07\tFood Court\t\u0633\u0627\u062d\u0629 \u0627\u0644\u0637\u0639\u0627\u0645\thttps://cenomi.com/events/seafood_week.jpg\n3\t4\tKids Fun Day\t\u064a\u0648\u0645 \u062a\u0631\u0641\u064a\u0647 \u0627\u0644\u0623\u0637\u0641al\tActivities and games for children of all ages.\t\u0623\u0646\u0634\u0637\u0629 \u0648\u0623\u0644\u0639\u0627\u0628 \u0644\u0644\u0623\u0637\u0641\u0627\u0644 \u0645\u0646 \u062c\u0645\u064a\u0639 \u0627\u0644\u0623\u0639\u0645\u0627\u0631.\t2025-03-30\t2025-03-30\tThird Floor Play Area\t\u0645\u0646\u0637\u0642\u0629 \u0627\u0644\u0644\u0639\u0628 \u0641\u064a \u0627\u0644\u0637\u0627\u0628\u0642 \u0627\u0644\u062b\u0627\u0644\u062b\thttps://cenomi.com/events/kids_fun_day.jpg\n*/\n\n\nCREATE TABLE loyalty_program (\n\tprogram_id SERIAL NOT NULL, \n\tname_en VARCHAR(100) NOT NULL, \n\tname_ar VARCHAR(100) NOT NULL, \n\tdescription_en TEXT NOT NULL, \n\tdescription_ar TEXT NOT NULL, \n\tpoints_to_currency_ratio NUMERIC(10, 2) NOT NULL, \n\tCONSTRAINT loyalty_program_pkey PRIMARY KEY (program_id)\n)\n\n/*\n3 rows from loyalty_program table:\nprogram_id\tname_en\tname_ar\tdescription_en\tdescription_ar\tpoints_to_currency_ratio\n\n*/\n\n\nCREATE TABLE mall_services (\n\tmall_id INTEGER NOT NULL, \n\tservice_id INTEGER NOT NULL, \n\tdetails_en VARCHAR(255), \n\tdetails_ar VARCHAR(255), \n\tlocation_in_mall_en VARCHAR(255), \n\tlocation_in_mall_ar VARCHAR(255), \n\tCONSTRAINT mall_services_pkey PRIMARY KEY (mall_id, service_id), \n\tCONSTRAINT mall_services_mall_id_fkey FOREIGN KEY(mall_id) REFERENCES malls (mall_id) ON DELETE CASCADE, \n\tCONSTRAINT mall_services_service_id_fkey FOREIGN KEY(service_id) REFERENCES services (service_id) ON DELETE CASCADE\n)\n\n/*\n3 rows from mall_services table:\nmall_id\tservice_id\tdetails_en\tdetails_ar\tlocation_in_mall_en\tlocation_in_mall_ar\n1\t1\t500 spaces available\t500 \u0645\u0643\u0627\u0646 \u0645\u062a\u0627\u062d\tBasement Level\t\u0627\u0644\u0637\u0627\u0628\u0642 \u0627\u0644\u0633\u0641\u0644\u064a\n1\t2\tSeparate areas for men and women\t\u0645\u0646\u0627\u0637\u0642 \u0645\u0646\u0641\u0635\u0644\u0629 \u0644\u0644\u0631\u062c\u0627\u0644 \u0648\u0627\u0644\u0646\u0633\u0627\u0621\tSecond Floor\t\u0627\u0644\u0637\u0627\u0628\u0642 \u0627\u0644\u062b\u0627\u0646\u064a\n1\t3\tCleaned hourly\t\u062a\u0646\u0638\u064a\u0641 \u0643\u0644 \u0633\u0627\u0639\u0629\tAll Floors\t\u062c\u0645\u064a\u0639 \u0627\u0644\u0637\u0648\u0627\u0628\u0642\n*/\n\n\nCREATE TABLE malls (\n\tmall_id SERIAL NOT NULL, \n\tname_en VARCHAR(100) NOT NULL, \n\tname_ar VARCHAR(100) NOT NULL, \n\tlocation_en TEXT NOT NULL, \n\tlocation_ar TEXT NOT NULL, \n\topening_hours_en VARCHAR(100) NOT NULL, \n\topening_hours_ar VARCHAR(100) NOT NULL, \n\tcontact_phone VARCHAR(20) NOT NULL, \n\temail VARCHAR(100) NOT NULL, \n\tgeo_latitude NUMERIC(10, 8) NOT NULL, \n\tgeo_longitude NUMERIC(11, 8) NOT NULL, \n\tCONSTRAINT malls_pkey PRIMARY KEY (mall_id)\n)\n\n/*\n3 rows from malls table:\nmall_id\tname_en\tname_ar\tlocation_en\tlocation_ar\topening_hours_en\topening_hours_ar\tcontact_phone\temail\tgeo_latitude\tgeo_longitude\n1\tRiyadh Season Mall\t\u0645\u0648\u0644 \u0645\u0648\u0633\u0645 \u0627\u0644\u0631\u064a\u0627\u0636\tRiyadh, Kingdom Centre District\t\u0627\u0644\u0631\u064a\u0627\u0636\u060c \u062d\u064a \u0645\u0631\u0643\u0632 \u0627\u0644\u0645\u0645\u0644\u0643\u0629\t10 AM - 11 PM\t10 \u0635\u0628\u0627\u062d\u064b\u0627 - 11 \u0645\u0633\u0627\u0621\u064b\t+966112345678\triyadhseason@cenomi.com\t24.71110000\t46.67440000\n2\tJeddah Waterfront Mall\t\u0645\u0648\u0644 \u062c\u062f\u0629 \u0648\u0627\u062c\u0647\u0629 \u0627\u0644\u0628\u062d\u0631\tJeddah, Al Hamra District\t\u062c\u062f\u0629\u060c \u062d\u064a \u0627\u0644\u062d\u0645\u0631\u0627\u0621\t10 AM - 12 AM\t10 \u0635\u0628\u0627\u062d\u064b\u0627 - 12 \u0635\u0628\u0627\u062d\u064b\u0627\t+966122345679\tjeddahwaterfront@cenomi.com\t21.52780000\t39.15580000\n3\tDammam Corniche Mall\t\u0645\u0648\u0644 \u0643\u0648\u0631\u0646\u064a\u0634 \u0627\u0644\u062f\u0645\u0627\u0645\tDammam, Al Shati District\t\u0627\u0644\u062f\u0645\u0627\u0645\u060c \u062d\u064a \u0627\u0644\u0634\u0627\u0637\u0626\t9 AM - 11 PM\t9 \u0635\u0628\u0627\u062d\u064b\u0627 - 11 \u0645\u0633\u0627\u0621\u064b\t+966132345680\tdammamcorniche@cenomi.com\t26.40150000\t50.16530000\n*/\n\n\nCREATE TABLE offers (\n\toffer_id SERIAL NOT NULL, \n\tstore_id INTEGER, \n\ttitle_en VARCHAR(100) NOT NULL, \n\ttitle_ar VARCHAR(100) NOT NULL, \n\tdescription_en TEXT NOT NULL, \n\tdescription_ar TEXT NOT NULL, \n\tdiscount_percentage INTEGER, \n\tstart_date DATE NOT NULL, \n\tend_date DATE NOT NULL, \n\tterms_conditions_en TEXT, \n\tterms_conditions_ar TEXT, \n\timage_url VARCHAR(255), \n\tCONSTRAINT offers_pkey PRIMARY KEY (offer_id), \n\tCONSTRAINT offers_store_id_fkey FOREIGN KEY(store_id) REFERENCES stores (store_id) ON DELETE CASCADE\n)\n\n/*\n3 rows from offers table:\noffer_id\tstore_id\ttitle_en\ttitle_ar\tdescription_en\tdescription_ar\tdiscount_percentage\tstart_date\tend_date\tterms_conditions_en\tterms_conditions_ar\timage_url\n2\t4\tH&M Clearance\t\u062a\u0635\u0641\u064a\u0629 \u0625\u062a\u0634 \u0622\u0646\u062f \u0625\u0645\t50% off selected items.\t\u062e\u0635\u0645 50% \u0639\u0644\u0649 \u0639\u0646\u0627\u0635\u0631 \u0645\u062e\u062a\u0627\u0631\u0629.\t50\t2025-04-01\t2025-04-15\tWhile stocks last.\t\u062d\u062a\u0649 \u0646\u0641\u0627\u062f \u0627\u0644\u0643\u0645\u064a\u0629.\thttps://cenomi.com/offers/hm_clearance.jpg\n3\t5\tDamas Ramadan Offer\t\u0639\u0631\u0636 \u0631\u0645\u0636\u0627\u0646 \u062f\u0627\u0645\u0627\u0633\t20% off gold jewelry.\t\u062e\u0635\u0645 20% \u0639\u0644\u0649 \u0627\u0644\u0645\u062c\u0648\u0647\u0631\u0627\u062a \u0627\u0644\u0630\u0647\u0628\u064a\u0629.\t20\t2025-03-20\t2025-04-20\tValid with purchase over SAR 1000.\t\u0635\u0627\u0644\u062d \u0645\u0639 \u0634\u0631\u0627\u0621 \u0623\u0643\u062b\u0631 \u0645\u0646 1000 \u0631\u064a\u0627\u0644.\thttps://cenomi.com/offers/damas_ramadan.jpg\n4\t7\tNike Summer Deal\t\u0639\u0631\u0636 \u0635\u064a\u0641\u064a \u0646\u0627\u064a\u0643\t25% off running shoes.\t\u062e\u0635\u0645 25% \u0639\u0644\u0649 \u0623\u062d\u0630\u064a\u0629 \u0627\u0644\u062c\u0631\u064a.\t25\t2025-06-01\t2025-06-30\tOnline and in-store.\t\u0623\u0648\u0646\u0644\u0627\u064a\u0646 \u0648\u0641\u064a \u0627\u0644\u0645\u062a\u062c\u0631.\thttps://cenomi.com/offers/nike_summer.jpg\n*/\n\n\nCREATE TABLE products (\n\tproduct_id SERIAL NOT NULL, \n\tstore_id INTEGER, \n\tname_en VARCHAR(100) NOT NULL, \n\tname_ar VARCHAR(100) NOT NULL, \n\tdescription_en TEXT, \n\tdescription_ar TEXT, \n\tprice NUMERIC(10, 2) NOT NULL, \n\timage_url VARCHAR(255), \n\tCONSTRAINT products_pkey PRIMARY KEY (product_id), \n\tCONSTRAINT products_store_id_fkey FOREIGN KEY(store_id) REFERENCES stores (store_id) ON DELETE CASCADE\n)\n\n/*\n3 rows from products table:\nproduct_id\tstore_id\tname_en\tname_ar\tdescription_en\tdescription_ar\tprice\timage_url\n1\t1\tZara Leather Jacket\t\u062c\u0627\u0643\u064a\u062a \u062c\u0644\u062f\u064a \u0632\u0627\u0631\u0627\tStylish black leather jacket.\t\u062c\u0627\u0643\u064a\u062a \u062c\u0644\u062f\u064a \u0623\u0633\u0648\u062f \u0623\u0646\u064a\u0642.\t299.00\thttps://cenomi.com/products/zara_leather.jpg\n2\t1\tZara Denim Jeans\t\u062c\u064a\u0646\u0632 \u062f\u064a\u0646\u0645 \u0632\u0627\u0631\u0627\tSlim-fit blue jeans.\t\u062c\u064a\u0646\u0632 \u0623\u0632\u0631\u0642 \u0636\u064a\u0642.\t149.00\thttps://cenomi.com/products/zara_jeans.jpg\n3\t2\tSamsung Galaxy S23\t\u0633\u0627\u0645\u0633\u0648\u0646\u062c \u062c\u0627\u0644\u0643\u0633\u064a S23\tLatest smartphone with 5G.\t\u0647\u0627\u062a\u0641 \u0630\u0643\u064a \u062d\u062f\u064a\u062b \u0645\u0639 5G.\t3499.00\thttps://cenomi.com/products/samsung_s23.jpg\n*/\n\n\nCREATE TABLE services (\n\tservice_id SERIAL NOT NULL, \n\tname_en VARCHAR(100) NOT NULL, \n\tname_ar VARCHAR(100) NOT NULL, \n\tdescription_en TEXT NOT NULL, \n\tdescription_ar TEXT NOT NULL, \n\tCONSTRAINT services_pkey PRIMARY KEY (service_id)\n)\n\n/*\n3 rows from services table:\nservice_id\tname_en\tname_ar\tdescription_en\tdescription_ar\n1\tParking\t\u0645\u0648\u0642\u0641 \u0633\u064a\u0627\u0631\u0627\u062a\tSecure parking facilities with ample spaces.\t\u0645\u0631\u0627\u0641\u0642 \u0645\u0648\u0627\u0642\u0641 \u0622\u0645\u0646\u0629 \u0645\u0639 \u0645\u0633\u0627\u062d\u0627\u062a \u0648\u0641\u064a\u0631\u0629.\n2\tPrayer Rooms\t\u063a\u0631\u0641 \u0627\u0644\u0635\u0644\u0627\u0629\tDedicated prayer rooms for men and women.\t\u063a\u0631\u0641 \u0635\u0644\u0627\u0629 \u0645\u062e\u0635\u0635\u0629 \u0644\u0644\u0631\u062c\u0627\u0644 \u0648\u0627\u0644\u0646\u0633\u0627\u0621.\n3\tRestrooms\t\u062f\u0648\u0631\u0627\u062a \u0627\u0644\u0645\u064a\u0627\u0647\tClean and accessible restrooms throughout the mall.\t\u062f\u0648\u0631\u0627\u062a \u0645\u064a\u0627\u0647 \u0646\u0638\u064a\u0641\u0629 \u0648\u0645\u062a\u0627\u062d\u0629 \u0641\u064a \u062c\u0645\u064a\u0639 \u0623\u0646\u062d\u0627\u0621 \u0627\u0644\u0645\u0648\u0644.\n*/\n\n\nCREATE TABLE store_categories (\n\tcategory_id SERIAL NOT NULL, \n\tname_en VARCHAR(100) NOT NULL, \n\tname_ar VARCHAR(100) NOT NULL, \n\tCONSTRAINT store_categories_pkey PRIMARY KEY (category_id)\n)\n\n/*\n3 rows from store_categories table:\ncategory_id\tname_en\tname_ar\n1\tFashion\t\u0623\u0632\u064a\u0627\u0621\n2\tElectronics\t\u0625\u0644\u0643\u062a\u0631\u0648\u0646\u064a\u0627\u062a\n3\tFood & Beverage\t\u0637\u0639\u0627\u0645 \u0648\u0645\u0634\u0631\u0648\u0628\u0627\u062a\n*/\n\n\nCREATE TABLE store_updates (\n\tupdate_id SERIAL NOT NULL, \n\tstore_id INTEGER, \n\ttenant_id INTEGER, \n\tupdate_type VARCHAR(50) NOT NULL, \n\tentity_id INTEGER, \n\tentity_type VARCHAR(50) NOT NULL, \n\told_values JSONB, \n\tnew_values JSONB, \n\tstatus VARCHAR(20) DEFAULT 'pending'::character varying, \n\tcreated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP, \n\tprocessed_at TIMESTAMP WITHOUT TIME ZONE, \n\tCONSTRAINT store_updates_pkey PRIMARY KEY (update_id), \n\tCONSTRAINT store_updates_store_id_fkey FOREIGN KEY(store_id) REFERENCES stores (store_id) ON DELETE CASCADE, \n\tCONSTRAINT store_updates_tenant_id_fkey FOREIGN KEY(tenant_id) REFERENCES tenant_users (user_id) ON DELETE SET NULL\n)\n\n/*\n3 rows from store_updates table:\nupdate_id\tstore_id\ttenant_id\tupdate_type\tentity_id\tentity_type\told_values\tnew_values\tstatus\tcreated_at\tprocessed_at\n\n*/\n\n\nCREATE TABLE stores (\n\tstore_id SERIAL NOT NULL, \n\tmall_id INTEGER, \n\tcategory_id INTEGER, \n\tname_en VARCHAR(100) NOT NULL, \n\tname_ar VARCHAR(100) NOT NULL, \n\tdescription_en TEXT NOT NULL, \n\tdescription_ar TEXT NOT NULL, \n\tlocation_in_mall_en VARCHAR(255) NOT NULL, \n\tlocation_in_mall_ar VARCHAR(255) NOT NULL, \n\topening_hours_en VARCHAR(100) NOT NULL, \n\topening_hours_ar VARCHAR(100) NOT NULL, \n\tcontact_phone VARCHAR(20), \n\temail VARCHAR(100), \n\tlogo_url VARCHAR(255), \n\tCONSTRAINT stores_pkey PRIMARY KEY (store_id), \n\tCONSTRAINT stores_category_id_fkey FOREIGN KEY(category_id) REFERENCES store_categories (category_id), \n\tCONSTRAINT stores_mall_id_fkey FOREIGN KEY(mall_id) REFERENCES malls (mall_id) ON DELETE CASCADE\n)\n\n/*\n3 rows from stores table:\nstore_id\tmall_id\tcategory_id\tname_en\tname_ar\tdescription_en\tdescription_ar\tlocation_in_mall_en\tlocation_in_mall_ar\topening_hours_en\topening_hours_ar\tcontact_phone\temail\tlogo_url\n1\t1\t1\tZara\t\u0632\u0627\u0631\u0627\tTrendy fashion for all ages.\t\u0623\u0632\u064a\u0627\u0621 \u0639\u0635\u0631\u064a\u0629 \u0644\u062c\u0645\u064a\u0639 \u0627\u0644\u0623\u0639\u0645\u0627\u0631.\tSecond Floor\t\u0627\u0644\u0637\u0627\u0628\u0642 \u0627\u0644\u062b\u0627\u0646\u064a\t10 AM - 11 PM\t10 \u0635\u0628\u0627\u062d\u064b\u0627 - 11 \u0645\u0633\u0627\u0621\u064b\t+966112345710\tzara.riyadh@cenomi.com\thttps://cenomi.com/logos/zara.png\n2\t1\t2\tSamsung\t\u0633\u0627\u0645\u0633\u0648\u0646\u062c\tLatest smartphones and electronics.\t\u0623\u062d\u062f\u062b \u0627\u0644\u0647\u0648\u0627\u062a\u0641 \u0627\u0644\u0630\u0643\u064a\u0629 \u0648\u0627\u0644\u0625\u0644\u0643\u062a\u0631\u0648\u0646\u064a\u0627\u062a.\tFirst Floor\t\u0627\u0644\u0637\u0627\u0628\u0642 \u0627\u0644\u0623\u0648\u0644\t10 AM - 11 PM\t10 \u0635\u0628\u0627\u062d\u064b\u0627 - 11 \u0645\u0633\u0627\u0621\u064b\t+966112345711\tsamsung.riyadh@cenomi.com\thttps://cenomi.com/logos/samsung.png\n3\t1\t3\tStarbucks\t\u0633\u062a\u0627\u0631\u0628\u0643\u0633\tPremium coffee and snacks.\t\u0642\u0647\u0648\u0629 \u0645\u0645\u064a\u0632\u0629 \u0648\u0648\u062c\u0628\u0627\u062a \u062e\u0641\u064a\u0641\u0629.\tGround Floor\t\u0627\u0644\u0637\u0627\u0628\u0642 \u0627\u0644\u0623\u0631\u0636\u064a\t10 AM - 11 PM\t10 \u0635\u0628\u0627\u062d\u064b\u0627 - 11 \u0645\u0633\u0627\u0621\u064b\t+966112345712\tstarbucks.riyadh@cenomi.com\thttps://cenomi.com/logos/starbucks.png\n*/\n\n\nCREATE TABLE tenant_users (\n\tuser_id SERIAL NOT NULL, \n\tstore_id INTEGER, \n\tusername VARCHAR(50) NOT NULL, \n\tpassword_hash VARCHAR(255) NOT NULL, \n\tfull_name VARCHAR(100) NOT NULL, \n\temail VARCHAR(100) NOT NULL, \n\tphone VARCHAR(20) NOT NULL, \n\trole VARCHAR(50) NOT NULL, \n\tlast_login TIMESTAMP WITHOUT TIME ZONE, \n\tCONSTRAINT tenant_users_pkey PRIMARY KEY (user_id), \n\tCONSTRAINT tenant_users_store_id_fkey FOREIGN KEY(store_id) REFERENCES stores (store_id) ON DELETE CASCADE, \n\tCONSTRAINT tenant_users_email_key UNIQUE (email), \n\tCONSTRAINT tenant_users_username_key UNIQUE (username)\n)\n\n/*\n3 rows from tenant_users table:\nuser_id\tstore_id\tusername\tpassword_hash\tfull_name\temail\tphone\trole\tlast_login\n\n*/"
        
        # Extract the SQL schema string
        if isinstance(raw_schema, dict) and "schema_info" in raw_schema:
            schema_sql = raw_schema["schema_info"]
        else:
            schema_sql = str(raw_schema)  # Fallback if structure is different
        
        # Parse SQL schema into structured format
        parsed_schema = parse_sql_schema(schema_sql)
        
        # Map entity type to table name (assuming plural table names)
        table_name = entity_type + "s"  # e.g., "offer" -> "offers"
        
        # Check if table exists in the schema
        if table_name in parsed_schema:
            table_schema = parsed_schema[table_name]
            required_fields = []
            
            # Extract required fields based on operation type
            if operation_type == "insert":
                # For inserts, we need all non-nullable fields except auto-increment columns
                required_fields = [
                    col_info["name"] for col_name, col_info in table_schema["columns"].items()
                    if not col_info["nullable"] and not col_info["auto_increment"]
                ]
            elif operation_type in ["update", "delete"]:
                # For updates and deletes, we need the primary key
                required_fields = [
                    col_info["name"] for col_name, col_info in table_schema["columns"].items()
                    if col_info["primary_key"]
                ]
                
                # For updates, typically we also need at least one field to update
                if operation_type == "update":
                    # Add some common fields that might be updated (adjust based on your actual schema)
                    common_update_fields = []
                    for field in ["name", "title", "description", "price", "discount", "status"]:
                        for col_name, col_info in table_schema["columns"]:
                            if field in col_name.lower():
                                common_update_fields.append(col_info["name"])
                                break
                    
                    # We'll suggest these fields but don't make them required
                    print(f"calculate_required_fields - Common fields that might be updated: {common_update_fields}")
            
            print(f"calculate_required_fields - Operation Type: {operation_type}, Entity: {entity_type}, Required Fields: {required_fields}")
            return required_fields
        else:
            print(f"calculate_required_fields - Schema not found for table: {table_name}. Returning empty required fields.")
            return []  # Table not found in schema
            
    except Exception as e:
        print(f"calculate_required_fields - Error processing schema: {e}. Returning empty required fields.")
        return []  # Error accessing/processing schema

# Example usage for testing
if __name__ == "__main__":
    
    # Test the function with different intents and entity types
    intents = [
        MockIntentCategory.TENANT_INSERT_OFFER,
        MockIntentCategory.TENANT_UPDATE_OFFER,
        MockIntentCategory.TENANT_DELETE_OFFER,
        MockIntentCategory.TENANT_INSERT_STORE,
        MockIntentCategory.TENANT_UPDATE_STORE
    ]
    
    entity_types = ["offer", "store"]
    
    print("===== TESTING REQUIRED FIELDS CALCULATION =====")
    
    for intent in intents:
        for entity_type in entity_types:
            if ("OFFER" in intent and entity_type == "offer") or ("STORE" in intent and entity_type == "store"):
                required_fields = calculate_required_fields(intent, entity_type)
                print(f"Intent: {intent}, Entity Type: {entity_type}")
                print(f"Required Fields: {required_fields}")
                print("-" * 50)
