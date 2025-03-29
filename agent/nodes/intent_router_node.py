from enum import Enum
from agent.agent_state import AgentState
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_loader import load_config

config = load_config()

class IntentCategory(str, Enum):
    """Intent categories for the IntentRouterNode."""
    # Customer Query Intents
    # Mall Information
    CUSTOMER_QUERY_MALL_INFO = "customer_query_mall_info"
    CUSTOMER_QUERY_MALL_HOURS = "customer_query_mall_hours"
    CUSTOMER_QUERY_MALL_CROWD = "customer_query_mall_crowd"
    LIST_MALLS = "list_malls"
    
    # Store Information
    CUSTOMER_QUERY_STORE_INFO = "customer_query_store_info"
    CUSTOMER_QUERY_SPECIFIC_STORE = "customer_query_specific_store"
    CUSTOMER_QUERY_STORE_CONTACT = "customer_query_store_contact"
    CUSTOMER_QUERY_STORE_CATEGORY = "customer_query_store_category"
    LIST_STORES_IN_MALL = "list_stores_in_mall"
    
    # Offer Information
    CUSTOMER_QUERY_OFFER_INFO = "customer_query_offer_info"
    CUSTOMER_QUERY_SPECIFIC_OFFER = "customer_query_specific_offer"
    CUSTOMER_QUERY_OFFER_LOYALTY = "customer_query_offer_loyalty"
    
    # Event Information
    CUSTOMER_QUERY_EVENT_INFO = "customer_query_event_info"
    CUSTOMER_QUERY_SPECIFIC_EVENT = "customer_query_specific_event"
    LIST_EVENTS_IN_MALL = "list_events_in_mall"
    
    # Amenity and Navigation
    CUSTOMER_QUERY_AMENITY_INFO = "customer_query_amenity_info"
    CUSTOMER_QUERY_NAVIGATION = "customer_query_navigation"
    CUSTOMER_QUERY_SERVICE_INFO = "customer_query_service_info"
    LIST_SERVICES_IN_MALL = "list_services_in_mall"
    
    # Food and Dining
    CUSTOMER_QUERY_RESTAURANT_INFO = "customer_query_restaurant_info"
    CUSTOMER_QUERY_SPECIFIC_RESTAURANT = "customer_query_specific_restaurant"
    
    # Loyalty Program
    CUSTOMER_QUERY_LOYALTY_INFO = "customer_query_loyalty_info"
    CUSTOMER_QUERY_LOYALTY_POINTS = "customer_query_loyalty_points"
    CUSTOMER_QUERY_LOYALTY_REDEMPTION = "customer_query_loyalty_redemption"
    
    # Recommendations, Assistance, Products, Temporal, General
    CUSTOMER_QUERY_RECOMMENDATION = "customer_query_recommendation"
    CUSTOMER_QUERY_ASSISTANCE = "customer_query_assistance"
    CUSTOMER_QUERY_PRODUCT_INFO = "customer_query_product_info"
    CUSTOMER_QUERY_TEMPORAL = "customer_query_temporal"
    CUSTOMER_QUERY_GENERAL = "customer_query_general"
    
    # Tenant Action Intents (Unchanged)
    TENANT_UPDATE_OFFER = "tenant_update_offer"
    TENANT_INSERT_OFFER = "tenant_insert_offer"
    TENANT_DELETE_OFFER = "tenant_delete_offer"
    TENANT_UPDATE_STORE = "tenant_update_store"
    TENANT_INSERT_STORE = "tenant_insert_store"
    TENANT_DELETE_STORE = "tenant_delete_store"
    TENANT_UPDATE_EVENT = "tenant_update_event"
    TENANT_INSERT_EVENT = "tenant_insert_event"
    TENANT_DELETE_EVENT = "tenant_delete_event"
    TENANT_ACTION = "tenant_action"
    
    # Other Intents
    GREETING = "greeting"
    POLITE_CLOSING = "polite_closing"
    OUT_OF_SCOPE = "out_of_scope"
    NOT_ALLOWED = "not_allowed"
    
# intent_llm = init_chat_model(model="llama3-70b-8192", model_provider="groq")
    # llm = init_chat_model(model="llama3-70b-8192", model_provider="groq")
intent_llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key)

def intent_router_node(state: AgentState) -> AgentState:
    """
    Intent Router Node: Classifies user intent and routes the conversation.
    """
    user_query = state["user_query"]
    conversation_history_str = state.get("conversation_history", "")
    role = state.get("role", "ANONYMOUS")
    mall_name = state.get("mall_name", "")
    
    intent_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
                You are an intent router for a chatbot system designed for Cenomi Malls, assisting customers and tenants.
                Classify user queries into the MOST appropriate intent category from the list below.

                Conversation History: {conversation_history_str}
                User Role: {user_role}
                Mall Name: {mall_name}

                Intent Categories:
                --- Customer Query Intents ---
                # Mall Information
                - customer_query_mall_info: General mall details (e.g., "Tell me about Cenomi Mall.")
                - customer_query_mall_hours: Mall operating hours (e.g., "What are the mall’s hours on Friday?")
                - customer_query_mall_crowd: Crowd levels (e.g., "Is the mall busy now?")
                - list_malls: List all malls (e.g., "What malls are there?")
                
                # Store Information
                - customer_query_store_info: General store info (e.g., "What stores are in this mall?")
                - customer_query_specific_store: Specific store details (e.g., "Where is Adidas?")
                - customer_query_store_contact: Store contact info (e.g., "What’s Pandora’s phone number?")
                - list_stores_in_mall: List stores in a mall (e.g., "List stores in Dubai Mall.")
                
                # Offer Information
                - customer_query_offer_info: General offers (e.g., "What offers are available?")
                - customer_query_specific_offer: Specific offers (e.g., "What promotions are at Carrefour?")
                - customer_query_offer_loyalty: Loyalty offers (e.g., "What offers do I get with loyalty points?")
                
                # Event Information
                - customer_query_event_info: General events (e.g., "What’s happening this weekend?")
                - customer_query_specific_event: Specific event details (e.g., "When is the fashion show?")
                - list_events_in_mall: List events (e.g., "List events in Mall X.")
                
                # Loyalty Program
                - customer_query_loyalty_info: Loyalty program details (e.g., "Tell me about the loyalty program.")
                - customer_query_loyalty_points: Points balance (e.g., "How many points do I have?")
                - customer_query_loyalty_redemption: Redemption options (e.g., "What can I redeem with my points?")
                
                # General
                - customer_query_general: Vague queries (e.g., "What’s good here?")
                
                --- Tenant Action Intents ---
                - tenant_update_offer: Update offer (e.g., "Update my offer details.")
                - tenant_insert_offer: Add offer (e.g., "Add a new offer.")
                - tenant_delete_offer: Delete offer (e.g., "Delete my offer.")
                - tenant_update_store: Update store (e.g., "Update my store hours.")
                - tenant_insert_store: Add store (e.g., "Register a new store.")
                - tenant_delete_store: Delete store (e.g., "Remove my store.")
                
                --- Other Intents ---
                - greeting: Greetings (e.g., "Hi there!")
                - polite_closing: Closings (e.g., "Thanks, bye!")
                - out_of_scope: Out-of-domain (e.g., "What’s the weather like?")
                - not_allowed: Not permitted (e.g., GUEST asking "How many points do I have?")

                Instructions:
                1. Analyze the query and classify it into the most specific intent.
                2. Prioritize 'greeting' and 'polite_closing' for conversational inputs.
                3. Use 'list_*' intents for list requests (SQL-based).
                4. For queries about loyalty points or programs (e.g., "my points", "redeem points", "loyalty program"):
                   - If role is GUEST, set intent to 'not_allowed'.
                   - If role is CUSTOMER or TENANT, classify as 'customer_query_loyalty_points', 'customer_query_loyalty_redemption', etc.
                5. For tenant CRUD actions (e.g., "update my offer", "add store", "delete event"):
                   - If role is TENANT, classify as 'tenant_update_offer', 'tenant_insert_store', etc.
                   - If role is GUEST or CUSTOMER, set intent to 'not_allowed'.
                6. Default to 'customer_query_general' for vague queries or 'out_of_scope' if unrelated to malls.
                7. Return ONLY the intent category name.
                """
            ),
            ("human", "{user_query}"),
        ]
    )
    
    intent_prompt = intent_prompt_template.format(
        user_query=user_query, 
        conversation_history_str=conversation_history_str, 
        user_role=role,
        mall_name=mall_name
    )
    intent_response = intent_llm.invoke(intent_prompt)
    intent_category_str = intent_response.content.strip()
    updated_state = state.copy()
    
    try:
        intent_category = IntentCategory(intent_category_str)
        if intent_category == IntentCategory.NOT_ALLOWED:
            updated_state["not_allowed"] = True
            next_node = "llm_call_node"
        elif intent_category in [IntentCategory.GREETING, IntentCategory.POLITE_CLOSING]:
            next_node = "llm_call_node"
        elif intent_category in [
            IntentCategory.CUSTOMER_QUERY_MALL_INFO,
            IntentCategory.CUSTOMER_QUERY_MALL_HOURS,
            IntentCategory.CUSTOMER_QUERY_MALL_CROWD,
            IntentCategory.LIST_MALLS,
            IntentCategory.CUSTOMER_QUERY_STORE_INFO,
            IntentCategory.CUSTOMER_QUERY_SPECIFIC_STORE,
            IntentCategory.CUSTOMER_QUERY_STORE_CONTACT,
            IntentCategory.CUSTOMER_QUERY_STORE_CATEGORY,
            IntentCategory.LIST_STORES_IN_MALL,
            IntentCategory.CUSTOMER_QUERY_OFFER_INFO,
            IntentCategory.CUSTOMER_QUERY_SPECIFIC_OFFER,
            IntentCategory.CUSTOMER_QUERY_OFFER_LOYALTY,
            IntentCategory.CUSTOMER_QUERY_EVENT_INFO,
            IntentCategory.CUSTOMER_QUERY_SPECIFIC_EVENT,
            IntentCategory.LIST_EVENTS_IN_MALL,
            IntentCategory.CUSTOMER_QUERY_AMENITY_INFO,
            IntentCategory.CUSTOMER_QUERY_NAVIGATION,
            IntentCategory.CUSTOMER_QUERY_SERVICE_INFO,
            IntentCategory.LIST_SERVICES_IN_MALL,
            IntentCategory.CUSTOMER_QUERY_RESTAURANT_INFO,
            IntentCategory.CUSTOMER_QUERY_SPECIFIC_RESTAURANT,
            IntentCategory.CUSTOMER_QUERY_LOYALTY_INFO,
            IntentCategory.CUSTOMER_QUERY_LOYALTY_POINTS,
            IntentCategory.CUSTOMER_QUERY_LOYALTY_REDEMPTION,
            IntentCategory.CUSTOMER_QUERY_PRODUCT_INFO,
            IntentCategory.CUSTOMER_QUERY_TEMPORAL,
        ]:
            next_node = "tool_selection_node"
        elif intent_category in [
            IntentCategory.CUSTOMER_QUERY_RECOMMENDATION,
            IntentCategory.CUSTOMER_QUERY_ASSISTANCE,
            IntentCategory.CUSTOMER_QUERY_GENERAL,
            IntentCategory.OUT_OF_SCOPE,
            
            IntentCategory.TENANT_UPDATE_OFFER,
            IntentCategory.TENANT_INSERT_OFFER,
            IntentCategory.TENANT_DELETE_OFFER,
            IntentCategory.TENANT_UPDATE_STORE,
            IntentCategory.TENANT_INSERT_STORE,
            IntentCategory.TENANT_DELETE_STORE,
            IntentCategory.TENANT_UPDATE_EVENT,
            IntentCategory.TENANT_INSERT_EVENT,
            IntentCategory.TENANT_DELETE_EVENT,
        ]:
            next_node = "llm_call_node"
        else:
            next_node = "tool_selection_node"
    except ValueError:
        intent_category = IntentCategory.OUT_OF_SCOPE
        next_node = "llm_call_node"
    
    updated_state["intent"] = intent_category
    updated_state["next_node"] = next_node
    return updated_state