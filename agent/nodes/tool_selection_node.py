from typing import Dict, Any, List
from agent.agent_state import AgentState
from agent.nodes.intent_router_node import IntentCategory
from enum import Enum
from agent.utils.relevance_utils import evaluate_relevance_function
from agent.tools.vector_db_search_tool import VectorDBSearchTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_loader import load_config

config = load_config()
# Define Tool Names
class ToolName(str, Enum):
    VECTOR_DB_SEARCH = "vector_db_search_tool"
    SQL_DATABASE_QUERY = "sql_database_query_tool"
    NO_TOOL = "no_tool"  # For direct LLM responses

# llm = init_chat_model(model="llama3-70b-8192", model_provider="groq")
llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key)
output_parser = StrOutputParser()

def tool_selection_node(state: AgentState) -> AgentState:
    """
    Intelligent Tool Selection Node: Routes vague queries to vector tool and specific queries to SQL tool.
    """
    print("--- Intelligent Tool Selection Node ---")
    user_query: str = state.get("user_query")
    intent: IntentCategory = state.get("intent")

    # 1. Direct routing for conversational intents
    if intent in [IntentCategory.GREETING, IntentCategory.POLITE_CLOSING]:
        return _prepare_response_state(state, [], {}, "llm_call_node")

    # 2. Direct routing for tenant actions
    if _is_tenant_action_intent(intent):
        return _prepare_response_state(state, [ToolName.SQL_DATABASE_QUERY.value], {}, "tool_invocation_node")

    # 3. For out-of-scope intents, go directly to LLM
    if intent == IntentCategory.OUT_OF_SCOPE:
        return _prepare_response_state(state, [], {}, "llm_call_node")

    # 4. Determine if query is specific or vague and route accordingly
    query_specificity = _analyze_query_specificity(user_query, intent)
    
    # 5. Route based on specificity
    if query_specificity['is_specific']:
        print("Query identified as specific. Routing to SQL Database Tool.")
        selected_tool = ToolName.SQL_DATABASE_QUERY.value
        next_node = "tool_invocation_node"
        tool_output = {}
    else:
        print("Query identified as vague. Routing to Vector DB Search Tool.")
        vector_db_tool = VectorDBSearchTool()
        vector_db_output = vector_db_tool.run(user_query)
        
        # Check if vector DB output is relevant enough
        relevance_score = evaluate_relevance_function(user_query, vector_db_output, intent)
        print(f"VectorDB relevance score: {relevance_score}")
        
        if relevance_score < 0.6:
            selected_tool = ToolName.SQL_DATABASE_QUERY.value
            next_node = "tool_invocation_node"
            tool_output = {}
        else:
            selected_tool = ToolName.VECTOR_DB_SEARCH.value
            next_node = "llm_call_node"
            tool_output = {ToolName.VECTOR_DB_SEARCH.value: vector_db_output}
    
    return _prepare_response_state(state, [selected_tool] if selected_tool != ToolName.NO_TOOL.value else [], tool_output, next_node)

def _is_tenant_action_intent(intent: IntentCategory) -> bool:
    """Check if the intent is related to tenant actions."""
    tenant_action_intents = [
        IntentCategory.TENANT_UPDATE_OFFER,
        IntentCategory.TENANT_INSERT_OFFER,
        IntentCategory.TENANT_DELETE_OFFER,
        IntentCategory.TENANT_UPDATE_STORE,
        IntentCategory.TENANT_INSERT_STORE,
        IntentCategory.TENANT_DELETE_STORE,
        IntentCategory.TENANT_UPDATE_EVENT,
        IntentCategory.TENANT_INSERT_EVENT,
        IntentCategory.TENANT_DELETE_EVENT,
    ]
    return intent in tenant_action_intents

def _analyze_query_specificity(query: str, intent: IntentCategory) -> Dict[str, bool]:
    """
    Analyze whether a query is specific (SQL) or vague (Vector DB).
    Returns a dictionary with analysis results.
    """
    query_lower = query.lower()
    
    # 1. Check if intent directly indicates a specific query
    specific_intents = [
        IntentCategory.LIST_MALLS,
        IntentCategory.LIST_STORES_IN_MALL,
        IntentCategory.LIST_SERVICES_IN_MALL,
        IntentCategory.LIST_EVENTS_IN_MALL,
        IntentCategory.CUSTOMER_QUERY_EVENT_INFO,
        IntentCategory.CUSTOMER_QUERY_SERVICE_QUERY
    ]
    
    intent_suggests_specific = intent in specific_intents
    
    # 2. Check for specific keywords/patterns
    # Indicators of specific queries (SQL-friendly)
    list_keywords = ["list", "show", "what are", "give me all", "show me all", "display all"]
    specific_entity_keywords = ["mall", "store", "event", "service", "offer", "brand", "promotion"]
    filter_keywords = ["where", "which", "with", "that have", "that are", "located in", "available at"]
    data_operation_keywords = ["update", "insert", "delete", "add", "remove", "change"]
    attribute_keywords = ["name", "location", "price", "time", "date", "phone", "address", "hours", "operating hours"]
    comparison_keywords = ["more than", "less than", "greater than", "before", "after", "between"]
    
    # Indicators of vague queries (Vector DB-friendly)
    vague_keywords = ["about", "information", "details", "tell me about", "what is", "how", "why", "recommend", "suggest", "best", "popular"]
    conceptual_keywords = ["experience", "atmosphere", "feel", "quality", "reputation", "review", "opinion", "think", "consider"]
    
    # Check for specific patterns
    has_list_keyword = any(keyword in query_lower for keyword in list_keywords)
    has_filter_keyword = any(keyword in query_lower for keyword in filter_keywords)
    has_specific_entity = any(keyword in query_lower for keyword in specific_entity_keywords)
    has_data_operation = any(keyword in query_lower for keyword in data_operation_keywords)
    has_attribute_keyword = any(keyword in query_lower for keyword in attribute_keywords)
    has_comparison = any(keyword in query_lower for keyword in comparison_keywords)
    
    # Check for vague patterns
    has_vague_keyword = any(keyword in query_lower for keyword in vague_keywords)
    has_conceptual_keyword = any(keyword in query_lower for keyword in conceptual_keywords)
    
    # 3. Check for structured query patterns
    has_structured_pattern = bool(re.search(r'\b(in|at|on|for|with|during|before|after)\b [\w\s]+', query_lower))
    
    # 4. Check for quantitative inquiries
    has_quantity_pattern = bool(re.search(r'\b(how many|count|total|number of)\b', query_lower))
    
    # 5. Check for entity specification
    has_named_entity = bool(re.search(r'"([^"]+)"|\'([^\']+)\'', query))  # Quoted entity names
    has_proper_noun = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query))  # Proper nouns
    
    # 6. Calculate specificity score
    specificity_indicators = [
        has_list_keyword,
        has_filter_keyword,
        has_specific_entity,
        has_data_operation,
        has_attribute_keyword,
        has_comparison,
        has_structured_pattern,
        has_quantity_pattern,
        has_named_entity,
        has_proper_noun,
        intent_suggests_specific
    ]
    
    vagueness_indicators = [
        has_vague_keyword,
        has_conceptual_keyword
    ]
    
    specificity_score = sum(1 for indicator in specificity_indicators if indicator)
    vagueness_score = sum(1 for indicator in vagueness_indicators if indicator)
    
    # Analyze the query text for specificity using LLM if needed
    llm_specificity_score = 0
    if abs(specificity_score - vagueness_score) <= 2:  # Close call, use LLM
        llm_specificity_score = _get_llm_specificity_score(query)
    
    # Calculate final specificity score considering all factors
    final_specificity_score = specificity_score - vagueness_score + llm_specificity_score
    
    # Determine if query is specific or vague
    is_specific = final_specificity_score > 0
    
    return {
        "is_specific": is_specific,
        "specificity_score": final_specificity_score,
        "has_list_keyword": has_list_keyword,
        "has_filter_keyword": has_filter_keyword,
        "has_specific_entity": has_specific_entity,
        "has_data_operation": has_data_operation,
        "has_attribute_keyword": has_attribute_keyword,
        "has_comparison": has_comparison,
        "has_structured_pattern": has_structured_pattern,
        "has_quantity_pattern": has_quantity_pattern,
        "has_named_entity": has_named_entity,
        "has_proper_noun": has_proper_noun,
        "has_vague_keyword": has_vague_keyword,
        "has_conceptual_keyword": has_conceptual_keyword,
        "intent_suggests_specific": intent_suggests_specific
    }

def _get_llm_specificity_score(query: str) -> int:
    """
    Use LLM to evaluate if a query is more specific or vague.
    Returns a score between -2 (very vague) and 2 (very specific).
    """
    specificity_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert at analyzing query specificity for database tools.
        
        Analyze if the following query is specific (concrete, well-defined, looking for precise information) 
        or vague (abstract, general, looking for conceptual information).
        
        Specific queries typically:
        - Ask for specific entities, attributes, or relationships
        - Contain filters, conditions, or constraints
        - Request lists, counts, or specific data points
        - Reference specific names, locations, times, or quantities
        
        Vague queries typically:
        - Ask for general information or concepts
        - Seek recommendations, opinions, or suggestions
        - Ask broad questions about topics
        - Don't specify exact criteria or filters
        
        Return a score from -2 to 2:
        -2: Very vague query
        -1: Somewhat vague query
         0: Neutral/balanced query
         1: Somewhat specific query
         2: Very specific query
        
        Return only the numeric score, nothing else.
        """),
        ("human", "{query}")
    ])
    
    try:
        specificity_chain = specificity_prompt | llm | output_parser
        score_text = specificity_chain.invoke({"query": query}).strip()
        score = int(score_text)
        # Ensure the score is within the expected range
        return max(-2, min(2, score))
    except Exception as e:
        print(f"Error getting LLM specificity score: {e}")
        return 0  # Neutral score if LLM fails

def _prepare_response_state(state: AgentState, selected_tools: List[str], tool_output: Dict, next_node: str) -> AgentState:
    """Prepare the response state with the selected tools and outputs."""
    updated_state: AgentState = state.copy()
    updated_state["selected_tools"] = selected_tools
    updated_state["tool_output"] = tool_output
    updated_state["next_node"] = next_node
    
    print("Tool Selection Node output:", {
        "selected_tools": selected_tools,
        "next_node": next_node
    })
    
    return updated_state