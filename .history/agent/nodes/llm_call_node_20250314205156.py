from typing import Any, Dict, List
from agent_state import AgentState
from nodes.intent_router_node import IntentCategory
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config

config = load_config()

def llm_call_node(state: AgentState) -> AgentState:
    """
    LLM Call Node: Generates natural language response using LLM based on agent state.
    """
    print("--- LLM Call Node ---")
    
    user_query = state["user_query"]
    intent = state["intent"]
    tool_output = state.get("tool_outputs", {})
    conversation_history = state.get("conversation_history", "")
    
    print(f"LLM Call Node - User Query: {user_query}")
    print(f"LLM Call Node - Intent: {intent}")
    print(f"LLM Call Node - Tool Output: {tool_output}")
    
    context_str = ""
    if tool_output:
        context_str = "Tool Output:\n"
        for tool_name, output in tool_output.items():
            context_str += f"{tool_name}:\n{output}\n"
    else:
        context_str += "No tool output generated.\n"
        
    llm = init_chat_model(model="llama3-8b-8192", model_provider="groq")
    output_parser = StrOutputParser()
    
    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
                You are Cenomi Chatbot, a funny and engaging chatbot assistant for Cenomi Malls.
                Your goal is to provide helpful, informative, and entertaining responses to users.
                You should use the context provided (user query, intent, tool outputs, conversation history) to generate your response.
                Be concise yet informative, and add a touch of humor and personality to make the conversation engaging and enjoyable.

                Context Information:
                --- START CONTEXT ---
                User Query: {user_query}
                Intent: {intent}
                Conversation History: {conversation_history}
                {context_information}  <-- Tool outputs or other relevant context will be inserted here
                --- END CONTEXT ---

                Response Guidelines:
                - Generate a natural, conversational, and user-friendly response.
                - Be concise and to the point, but provide enough detail to be helpful.
                - Incorporate humor and a friendly tone to make the chatbot more engaging.
                - If tool output is available, use it to provide specific and accurate information related to the user query.
                - If no tool output is available, generate a response based on the intent and conversation history, acknowledging the lack of specific information if necessary.
                - If the intent is 'out_of_scope', respond politely that you are designed to answer queries about malls, stores, offers, and events and cannot assist with out-of-scope queries.
                - Keep responses relatively short and avoid overly lengthy or complex sentences.
                - End your responses in a way that encourages further interaction (e.g., asking if the user has more questions).

                Generate Response:
                """
            ),
            ("human", "{user_query}"),
        ]
    )
    
    response_generation_chain = response_prompt | llm | output_parser
    
    try:
        llm_response_text = response_generation_chain.invoke({
            "user_query": user_query,
            "intent": intent,
            "conversation_history": conversation_history,
            "context_information": context_str
        })
        print("LLM Call Node - Generated Response:\n", llm_response_text)
        updated_state: AgentState = state.copy()
        updated_state["response"] = llm_response_text
        updated_state["next_node"] = "output_node"
        return updated_state

    except Exception as e:
        error_message = f"Error generating LLM response: {e}"
        print(error_message)
        updated_state: AgentState = state.copy()
        updated_state["response"] = "sorry, I am having trouble understanding you right now. Please try again later."
        updated_state["next_node"] = "output_node"
        return updated_state
