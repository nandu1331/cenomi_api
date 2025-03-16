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
    CUSTOMER_QUERY_SERVICE_QUERY = "customer_query_service_query"
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
    
    intent_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
                You are an intent router for a chatbot system designed for Cenomi Malls.
                Your task is to classify user queries into predefined intent categories.
                Analyze the user query and determine the MOST appropriate intent category from the list provided below.

                Intent Categories:
                --- Data/Tool Driven Intents (Mall Information) ---
                - customer_query: General customer questions about malls, stores, brands, offers, events, etc.
                - customer_query_mall_info: User is asking for information about a specific mall.
                - customer_query_brand_info: User is asking for information about a specific brand.
                - customer_query_offer_info: User is asking about offers generally or specifically.
                - customer_query_event_info: User is asking about event information.
                - customer_query_store_query: User is asking about store information.
                - customer_query_specific_store_query: User is asking about a specific store.
                - customer_query_service_query: User is asking about services at a mall.
                - list_malls: User explicitly wants a list of malls.
                - list_stores_in_mall: User wants a list of stores in a mall.
                - list_services_in_mall: User wants a list of services in a mall.
                - list_events_in_mall: User wants a list of events in a mall.

                --- Conversational Intents (Basic Chat) ---
                - greeting: User is greeting the chatbot (e.g., "Hi", "Hello", "Good morning").  Respond conversationally.
                - polite_closing: User is expressing thanks or closing the conversation politely (e.g., "Thank you", "Thanks", "Bye"). Respond with a polite closing.

                --- Other Intents ---
                - tenant_action: User query is related to tenant/mall staff actions.
                - out_of_scope: User query is outside the scope of mall-related information.

                Instructions:
                1. Carefully analyze the user query to understand the user's intent.
                2. Match the user query to the MOST appropriate intent category from the list above.
                3. Prioritize 'greeting' and 'polite_closing' for simple conversational inputs. If the user is just saying hello or goodbye, classify accordingly.
                4. For queries clearly asking for LISTS of malls, stores, services, or events, choose 'list_' intents (SQL required).
                5. For general information queries about malls, brands, offers, etc., choose 'customer_query' or 'customer_query_*' intents (VectorDB/SQL might be used).
                6. Classify queries related to tenant/mall actions as 'tenant_action'.
                7. Classify queries outside the mall domain as 'out_of_scope'.
                8. Return ONLY the intent category name as plain text. No explanations or comments.
                9. If unsure, default to 'customer_query' or 'out_of_scope' if truly unclear.

                Example User Queries and Intent Categories:
                User Query: "Hi"
                Intent Category: greeting

                User Query: "Hello there!"
                Intent Category: greeting

                User Query: "Good morning"
                Intent Category: greeting

                User Query: "Thank you for your help"
                Intent Category: polite_closing

                User Query: "Thanks, bye"
                Intent Category: polite_closing

                User Query: "Okay, goodbye"
                Intent Category: polite_closing

                User Query: "What are the events happening at Dubai Mall?"
                Intent Category: list_events_in_mall

                User Query: "What is the weather in Dubai?"
                Intent Category: out_of_scope

                User Query: "Hi, what are the offers today?"
                Intent Category: customer_query_offer_info # Still offer info even with greeting, information intent is stronger


                Classify the intent for the following user query:
                """),
            ("human", "{user_query}"),
        ]
    )
    
    intent_prompt = intent_prompt_template.format(user_query=user_query)
    
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
            IntentCategory.CUSTOMER_QUERY_SERVICE_QUERY,
            IntentCategory.LIST_MALLS,
            IntentCategory.LIST_STORES_IN_MALL,
            IntentCategory.LIST_SERVICES_IN_MALL,
            IntentCategory.LIST_EVENTS_IN_MALL,
        ]:
            next_node = "tool_selection_node"
        elif intent_category in [
            IntentCategory.GREETING, # New: Conversational Intents Route Directly to LLM
            IntentCategory.POLITE_CLOSING
        ]:
            print(f"Routing directly to LLM call node for conversational intent: {intent_category}")
            return "intent_router_conversational"
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