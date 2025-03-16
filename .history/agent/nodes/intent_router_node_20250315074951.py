from typing import Dict, Any, List
from enum import Enum
from langchain.chat_models import init_chat_model
from agent_state import AgentState
from langchain.prompts import ChatPromptTemplate

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
    LIST_MALLS = "list_malls"
    LIST_STORES_IN_MALL = "list_stores_in_mall"
    LIST_SERVICES_IN_MALL = "list_services_in_mall"
    LIST_EVENTS_IN_MALL = "list_events_in_mall" 
    
intent_llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")

def intent_router_node(state: AgentState) -> AgentState:
    """
    Intent Router Node: Classifies user intent and routes the conversation.
    """
    user_query = state["user_query"]
    conversation_history_str = state.get("conversation_history", "")
    
    print(f"User query for Intent Routing: {user_query}")
    
    intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """
            You are an intent router for a chatbot system designed for Cenomi Malls.
            Your task is to classify user queries into predefined intent categories.
            Analyze the user query and determine the most appropriate intent category from the list provided below.

            Intent Categories:
            - customer_query: General customer questions about malls, stores, brands, offers, events, etc. Use for broad informational queries.  Vector database search is often suitable for this intent.
            - customer_query_mall_info: User is asking for information about a specific mall (e.g., "Dubai Mall information"). Vector database search can be helpful here.
            - customer_query_brand_info: User is asking for information about a specific brand (e.g., "Nike brand details"). Vector database search is relevant.
            - customer_query_offer_info: User is asking about offers generally or for a specific category/brand (e.g., "What offers are available?", "Offers on shoes"). Vector database search is useful.
            - customer_query_event_info: User is asking about general event information or events related to a category (e.g., "What events are happening?"). SQL database might be suitable for listing events.
            - customer_query_store_query: User is asking about store information generally or in a category (e.g., "Find electronics stores"). Vector database and SQL might be used.
            - customer_query_specific_store_query: User is asking about a specific store (e.g., "Tell me about Apple store in Dubai Mall"). Vector database is often helpful.
            - customer_query_service_query: User is asking about services offered at a mall (e.g., "What services are at Mall of Emirates?"). SQL database preferred for listing.
            - list_malls: User explicitly wants a list of malls (e.g., "List all malls", "Show me malls in Dubai"). SQL database is REQUIRED for this intent.
            - list_stores_in_mall: User wants a list of stores in a specific mall (e.g., "List stores in City Centre Mirdif"). SQL database is REQUIRED.
            - list_services_in_mall: User wants a list of services in a specific mall (e.g., "What services does Dubai Mall have?"). SQL database is REQUIRED.
            - list_events_in_mall: User wants a list of events in a specific mall (e.g., "List events at Mall of Emirates"). SQL database is REQUIRED.

            - tenant_action: User query is related to tenant/mall staff actions (e.g., updating information - not yet implemented, future feature).
            - out_of_scope: User query is outside the scope of mall information, stores, offers, events, or services (e.g., "What is the weather?", "Tell me a joke").

            Instructions:
            1. Carefully analyze the user query to understand the user's intent.
            2. Match the user query to the MOST appropriate intent category from the list above.
            3. If the query clearly asks for a LIST of malls, stores, services, or events, choose the corresponding 'list_' intent (list_malls, list_stores_in_mall, etc.).  These intents REQUIRE SQL database access for accurate listing.
            4. For general customer questions, informational queries about malls, brands, offers, events, stores, or services that are NOT explicit list requests, choose 'customer_query' or the more specific 'customer_query_*' intents. Vector database search is often suitable for these.
            5. If the query is clearly outside the domain of mall-related information, classify it as 'out_of_scope'.
            6. For now, if the query seems related to tenant/mall staff actions, classify as 'tenant_action'.
            7. Return ONLY the intent category name as plain text (e.g., 'customer_query', 'list_malls', 'out_of_scope'). Do not include any explanations or comments.
            8. If you are unsure or cannot confidently classify the intent, default to 'customer_query' as a general fallback.

            Example User Queries and Intent Categories:
            User Query: "What are the events happening at Dubai Mall?"
            Intent Category: list_events_in_mall

            User Query: "List all malls in Dubai"
            Intent Category: list_malls

            User Query: "Find stores in Mall of Emirates that sell electronics"
            Intent Category: list_stores_in_mall  (Could also be customer_query_store_query - list intent is slightly more specific here)

            User Query: "What services are offered at City Centre Deira?"
            Intent Category: list_services_in_mall

            User Query: "Tell me about offers on shoes."
            Intent Category: customer_query_offer_info

            User Query: "Operating hours of Dubai Mall"
            Intent Category: customer_query_mall_info

            User Query: "Where is the nearest Starbucks?"
            Intent Category: customer_query_store_query

            User Query: "Who owns Nike brand?"
            Intent Category: customer_query_brand_info

            User Query: "Tell me about upcoming events."
            Intent Category: customer_query_event_info

            User Query: "What kind of stores are there?"
            Intent Category: customer_query_store_query

            User Query: "I want to update store hours"
            Intent Category: tenant_action

            User Query: "What is the weather in Dubai?"
            Intent Category: out_of_scope


            Classify the intent for the following user query:
            """),
        ("human", "{user_query}"),
    ]
)
    
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
            IntentCategory.LIST_MALLS,
            IntentCategory.LIST_STORES_IN_MALL,
            IntentCategory.LIST_SERVICES_IN_MALL,
            IntentCategory.LIST_EVENTS_IN_MALL,
        ]:
            next_node = "tool_selection_node"
        elif intent_category == IntentCategory.TENANT_ACTION:
            next_node = "llm_call_node"
        elif intent_category == IntentCategory.OUT_OF_SCOPE:
            next_node = "llm_call_node"
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