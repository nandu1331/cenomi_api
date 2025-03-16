from typing import Dict, Any, List
from enum import Enum
from langchain.chat_models import init_chat_model
from agent.agent_state import AgentState
from langchain.prompts import ChatPromptTemplate

class IntentCategory(str, Enum):
    """Intent categories for the IntentRouterNode."""
    print("--- Intent Router Node ---")
    CUSTOMER_QUERY = "customer_query"
    TENANT_ACTION = "tenant_action"
    OUT_OF_SCOPE = "out_of_scope"
    
    """Customer Query Intents"""
    CUSTOMER_QUERY_MALL_INFO = "customer_query_mall_info"
    CUSTOMER_QUERY_BRAND_INFO = "customer_query_brand_info"
    CUSTOMER_QUERY_OFFER_INFO = "customer_query_offer_info"
    CUSTOMER_QUERY_EVENT_INFO = "customer_query_event_info"
    CUSTOMER_QUERY_STORE_QUERY = "customer_query_store_query"
    CUSTOMER_QUERY_SPECIFIC_STORE_QUERY = "customer_query_specific_store_query"
    CUSTOMER_QUERY_SERVICE_QUERY = "customer_query_service_query"
    LIST_MALLS = "list_malls"
    LIST_STORES_IN_MALL = "list_stores_in_mall"
    LIST_SERVICES_IN_MALL = "list_services_in_mall"
    LIST_EVENTS_IN_MALL = "list_events_in_mall" 
    GREETING = "greeting"
    POLITE_CLOSING = "polite_closing"
    
    """Tenant Action Intents"""
    TENANT_UPDATE_OFFER = "tenant_update_offer"
    TENANT_INSERT_OFFER = "tenant_insert_offer"
    TENANT_DELETE_OFFER = "tenant_delete_offer"
    TENANT_UPDATE_STORE = "tenant_update_store"
    TENANT_INSERT_STORE = "tenant_insert_store"
    TENANT_DELETE_STORE = "tenant_delete_store"
    TENANT_UPDATE_EVENT = "tenant_update_event"
    TENANT_INSERT_EVENT = "tenant_insert_event"
    TENANT_DELETE_EVENT = "tenant_delete_event"
    
intent_llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")

def intent_router_node(state: AgentState) -> AgentState:
    """
    Intent Router Node: Classifies user intent and routes the conversation.
    """
    user_query = state["user_query"]
    conversation_history_str = state.get("conversation_history", "")
    
    print(f"User query for Intent Routing: {user_query}")
    
    intent_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
                You are an intent router for a chatbot system designed for Cenomi Malls, assisting both customers and mall tenants.
                Your task is to classify user queries into predefined intent categories.
                Analyze the user query and determine the MOST appropriate intent category from the list provided below.
                
                conversation history: {conversation_history_str}
                
                Intent Categories:

                --- Customer Query Intents (Mall Information) --- (Keep existing customer intents)
                - customer_query: General customer questions about malls, stores, brands, offers, events, etc.
                - customer_query_mall_info: User is asking for mall information.
                - customer_query_brand_info: User is asking for brand information.
                - customer_query_offer_info: User is asking about offers.
                - customer_query_event_info: User is asking about events.
                - customer_query_store_query: User is asking about store information.
                - customer_query_specific_store_query: User is asking about a specific store.
                - customer_query_service_query: User is asking about services at a mall.
                - list_malls: User wants a list of malls.
                - list_stores_in_mall: User wants a list of stores in a mall.
                - list_services_in_mall: User wants a list of services in a mall.
                - list_events_in_mall: User wants a list of events in a mall.
                - greeting: User is greeting the chatbot.
                - polite_closing: User is expressing thanks or closing politely.

                --- Tenant Action Intents (Data Manipulation - NEW SECTION) ---
                - tenant_update_offer: Tenant wants to update an existing offer (e.g., "update offer", "modify offer").
                - tenant_insert_offer: Tenant wants to create a new offer (e.g., "add new offer", "create offer").
                - tenant_delete_offer: Tenant wants to delete an offer (e.g., "remove offer", "delete offer").
                - tenant_update_store: Tenant wants to update store information (e.g., "update my store details").
                - tenant_insert_store: Tenant wants to add a new store (e.g., "register a new store").
                - tenant_delete_store: Tenant wants to delete a store (e.g., "remove my store").
                - tenant_update_event: Tenant wants to update event information (e.g., "change event details").
                - tenant_insert_event: Tenant wants to create a new event (e.g., "schedule an event").
                - tenant_delete_event: Tenant wants to delete an event (e.g., "cancel an event").

                --- Other Intents --- (Existing)
                - tenant_action: General tenant-related action (fallback for more complex tenant requests).
                - out_of_scope: Query is outside the scope of mall information and tenant actions.

                Instructions:
                1. Analyze the user query to understand the intent.
                2. Classify the query into the MOST appropriate intent category.
                3. Prioritize 'greeting' and 'polite_closing' for conversational inputs.
                4. For list requests, choose 'list_' intents (SQL required).
                5. For general mall info queries, use 'customer_query' or 'customer_query_*'.
                6. **For queries indicating tenants wanting to UPDATE, INSERT, or DELETE data related to offers, stores, events, etc., classify them into the corresponding 'tenant_update_*', 'tenant_insert_*', or 'tenant_delete_*' intents.**  Look for keywords like "update," "modify," "change," "add," "create," "new," "delete," "remove," along with entity names like "offer," "store," "event."
                7. Classify general tenant-related queries (not specific CRUD) as 'tenant_action'.
                8. Classify out-of-domain queries as 'out_of_scope'.
                9. Return ONLY the intent category name. No explanations.
                10. Default to 'customer_query' or 'out_of_scope' if unsure.

                Example User Queries and Intent Categories (Tenant Actions):
                User Query: "I want to update an offer"
                Intent Category: tenant_update_offer

                User Query: "Modify the discount on my summer sale offer"
                Intent Category: tenant_update_offer

                User Query: "Add a new offer for Ramadan"
                Intent Category: tenant_insert_offer

                User Query: "Create a store for my brand"
                Intent Category: tenant_insert_store

                User Query: "Delete the 'Flash Sale' offer"
                Intent Category: tenant_delete_offer

                User Query: "Change the timings of the New Year Event"
                Intent Category: tenant_update_event

                Classify the intent for the following user query:
                """),
            ("human", "{user_query}"),
        ]
    )
    
    intent_prompt = intent_prompt_template.format(user_query=user_query, conversation_history_str=conversation_history_str)
    
    intent_response = intent_llm.invoke(intent_prompt)
    intent_category_str = intent_response.content.strip()
    updated_state = AgentState = state.copy()
    
    print(f"Intent LLM Response: {intent_response.content}")
    print(f"Extracted Intent Category String: {intent_category_str}")
    
    try:
        intent_category = IntentCategory(intent_category_str)
        print(f"Parsed Intent Category: {intent_category}")
        if intent_category in [
            IntentCategory.GREETING,
            IntentCategory.POLITE_CLOSING
        ]:
            next_node = "llm_call_node"
        elif intent_category in [
            IntentCategory.CUSTOMER_QUERY,
            IntentCategory.CUSTOMER_QUERY_MALL_INFO,
            IntentCategory.CUSTOMER_QUERY_BRAND_INFO,
            IntentCategory.CUSTOMER_QUERY_OFFER_INFO,
            IntentCategory.CUSTOMER_QUERY_EVENT_INFO,
            IntentCategory.CUSTOMER_QUERY_STORE_QUERY,
            IntentCategory.CUSTOMER_QUERY_SPECIFIC_STORE_QUERY,
            IntentCategory.CUSTOMER_QUERY_SERVICE_QUERY,
            IntentCategory.LIST_MALLS,
            IntentCategory.LIST_STORES_IN_MALL,
            IntentCategory.LIST_SERVICES_IN_MALL,
            IntentCategory.LIST_EVENTS_IN_MALL,
        ]:
            next_node = "tool_selection_node"
        elif intent_category in [
            IntentCategory.TENANT_UPDATE_OFFER, # Route for Tenant Offer Updates
            IntentCategory.TENANT_INSERT_OFFER, # Route for Tenant Offer Insertions
            IntentCategory.TENANT_DELETE_OFFER, # Route for Tenant Offer Deletions
            IntentCategory.TENANT_UPDATE_STORE, # Example - Add routes for other entities as needed
            IntentCategory.TENANT_INSERT_STORE,
            IntentCategory.TENANT_DELETE_STORE,
            IntentCategory.TENANT_UPDATE_EVENT,
            IntentCategory.TENANT_INSERT_EVENT,
            IntentCategory.TENANT_DELETE_EVENT,
        ]:
            print(f"Routing to tenant action node for intent: {intent_category}")
            updated_state["tenant_main_query"] = user_query
            next_node = "tenant_action_node"
        elif intent_category == IntentCategory.TENANT_ACTION:
            next_node = "tool_selection_node"
        elif intent_category == IntentCategory.OUT_OF_SCOPE:
            next_node = "llm_call_node"
        else:
            next_node = "output_node"

        print(f"Routing to next node: {next_node}")
        
    except ValueError as e:
        print(f"Error parsing intent category: {e}. LLM returned invalid intent string: {intent_category_str}")
        intent_category = IntentCategory.OUT_OF_SCOPE
        next_node = "output_node"
        
    updated_state["intent"] = intent_category
    updated_state["next_node"] = next_node
    
    print("Intent Router Node State (Updated):", updated_state)
    return updated_state