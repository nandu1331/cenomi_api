from typing import Dict, Any, List
from enum import Enum
from langchain.chat_models import init_chat_model
from agent_state import AgentState

class IntentCategory(str, Enum):
    """Intent categories for the IntentRouterNode."""
    print("--- Intent Router Node ---")
    CUSTOMER_QUERY = "customer_query"
    TENANT_ACTION = "tenant_action"
    OUT_OF_SCOPE = "out_of_scope"
    
    CUSTOMER_QUERY_MALL_INFO = "customer_query_mall_info"
    CUSTOMER_QUERY_BRAND_INFO = "customer_query_brand_info"
    CUSTOMER_QUERY_OFFER_INFO = "customer_query_offer_info"
    CUSTOMER_QUERY_EVENT_INFO = "customer_query_event_info"
    CUSTOMER_QUERY_STORE_QUERY = "customer_query_store_query"
    CUSTOMER_QUERY_SPECIFIC_STORE_QUERY = "customer_query_specific_store_query"
    CUSTOMER_QUERY_LIST_MALLS = "customer_query_list_malls"
    CUSTOMER_QUERY_LIST_STORES = "customer_query_list_stores"
    CUSTOMER_QUERY_MALL_SERVICES = "customer_query_mall_services"
    CUSTOMER_QUERY_STORE_DETAILS = "customer_query_store_details"
    
intent_llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")

def intent_router_node(state: AgentState) -> AgentState:
    """
    Intent Router Node: Classifies user intent and routes the conversation.
    """
    user_query = state["user_query"]
    conversation_history_str = state.get("conversation_history", "")
    
    print(f"User query for Intent Routing: {user_query}")
    
    intent_prompt = f"""
    <<BEGIN PROMPT>>
    You are an intent classifier for a chatbot designed for Cenomi Malls.
    Your task is to classify the user's query into one of the following categories:

    Intent Categories:
    - {IntentCategory.CUSTOMER_QUERY.value}: General customer queries about malls, brands, offers, events, stores.
        - Subcategories (within customer_query):
            - {IntentCategory.CUSTOMER_QUERY_MALL_INFO.value}: Queries about mall information (e.g., "mall timings", "mall facilities", "what malls are there").
            - {IntentCategory.CUSTOMER_QUERY_BRAND_INFO.value}: Queries about brand information (e.g., "tell me about Nike", "what brands are in Dubai Mall").
            - {IntentCategory.CUSTOMER_QUERY_OFFER_INFO.value}: Queries about offers and promotions (e.g., "any offers at Mall of Emirates?", "discount on shoes").
            - {IntentCategory.CUSTOMER_QUERY_EVENT_INFO.value}: Queries about events (e.g., "events this weekend", "kids events in City Centre").
            - {IntentCategory.CUSTOMER_QUERY_STORE_QUERY.value}: Queries about store locations in general (e.g., "where is Apple store?", "find electronics stores").
            - {IntentCategory.CUSTOMER_QUERY_SPECIFIC_STORE_QUERY.value}: Queries about whether a specific brand has a store in a specific mall (e.g., "does Nike have a store in Dubai Mall?").
            - {IntentCategory.CUSTOMER_QUERY_LIST_MALLS.value}: Queries requesting a list of malls (e.g., "List all malls in Dubai").
            - {IntentCategory.CUSTOMER_QUERY_LIST_STORES.value}: Queries requesting a list of stores in a mall (e.g., "List the stores in City Centre Mall").
            - {IntentCategory.CUSTOMER_QUERY_MALL_SERVICES.value}: Inquiries about services offered by a mall (e.g., "What services does Mall of Emirates provide?").
    - {IntentCategory.TENANT_ACTION.value}: Actions initiated by mall tenants (e.g., updating offer information).
    - {IntentCategory.OUT_OF_SCOPE.value}: Queries that are outside the scope of the chatbot (e.g., "what's the weather?", chit-chat).

    Conversation History (Hybrid Context, if available):\n{conversation_history_str}\n

    User Query: {user_query}

    Classify the user query into ONE of the categories listed above.  Return ONLY the category name as a plain text string, nothing else. 
    For example, if the query is about mall timings, you should return: {IntentCategory.CUSTOMER_QUERY_MALL_INFO.value}
    If the query is about updating an offer, return: {IntentCategory.TENANT_ACTION.value}
    If the query is unrelated to malls, brands, offers, or events, return: {IntentCategory.OUT_OF_SCOPE.value}

    Classification:
    <<END PROMPT>>
    """
    
    print("Intent Classification Prompt:\n", intent_prompt)
    
    intent_response = intent_llm.invoke(intent_prompt)
    intent_category_str = intent_response.content.strip()
    
    print(f"Intent LLM Response: {intent_response.content}")
    print(f"Extracted Intent Category String: {intent_category_str}")
    
    try:
        intent_category = IntentCategory(intent_category_str)
        print(f"Parsed Intent Category: {intent_category}")
        
        if intent_category in [
            IntentCategory.CUSTOMER_QUERY,
            IntentCategory.CUSTOMER_QUERY_MALL_INFO,
            IntentCategory.CUSTOMER_QUERY_BRAND_INFO,
            IntentCategory.CUSTOMER_QUERY_OFFER_INFO,
            IntentCategory.CUSTOMER_QUERY_EVENT_INFO,
            IntentCategory.CUSTOMER_QUERY_STORE_QUERY,
            IntentCategory.CUSTOMER_QUERY_SPECIFIC_STORE_QUERY,
        ]:
            next_node = "tool_selection_node"
        elif intent_category == IntentCategory.TENANT_ACTION:
            next_node = "output_node"
        elif intent_category == IntentCategory.OUT_OF_SCOPE:
            next_node = "output_node"
        else:
            next_node = "output_node"

        print(f"Routing to next node: {next_node}")
        
    except ValueError as e:
        print(f"Error parsing intent category: {e}. LLM returned invalid intent string: {intent_category_str}")
        intent_category = IntentCategory.OUT_OF_SCOPE
        next_node = "output_node"
        
    updated_state = AgentState = state.copy()
    updated_state["intent"] = intent_category
    updated_state["next_node"] = next_node
    
    print("Intent Router Node State (Updated):", updated_state)
    return updated_state