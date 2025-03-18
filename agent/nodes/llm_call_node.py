from agent.agent_state import AgentState
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_loader import load_config

config = load_config()

llm = ChatGoogleGenerativeAI(model=config.llm.model_name, google_api_key=config.llm.api_key)
output_parser = StrOutputParser()

def llm_call_node(state: AgentState) -> AgentState:
    """
    LLM Call Node: Generates natural language response using LLM based on agent state.
    """
    print("--- LLM Call Node ---")
    
    user_query = state["user_query"]
    intent = state["intent"]
    tool_output = state.get("tool_outputs", {})
    
    # Convert conversation history from list of dicts to a formatted string
    conversation_history_list = state.get("conversation_history", [])
    conversation_history_str = "\n".join(
        [f"User: {turn['user']}\nAssistant: {turn['bot']}" 
         for turn in conversation_history_list if 'user' in turn and 'bot' in turn]
    ) or "No previous conversation history available."

    # Format tool output for context
    context_str = ""
    if tool_output:
        context_str = "Tool Output:\n"
        for tool_name, output in tool_output.items():
            context_str += f"{tool_name}:\n{output}\n"
    else:
        context_str += "No tool output generated.\n"

    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
                You are Cenomi Chatbot, a funny and engaging chatbot assistant for Cenomi Malls.
                Your goal is to provide helpful, informative, and entertaining responses to users.
                Use the provided context—including the user query, intent, conversation history, and any tool outputs—to craft your answer.

                Guidelines:
                1. If tool output is provided:
                    - **Structured SQL Results (e.g., lists of malls, stores, services, events):**
                      * Present the complete list of records.
                      * Use bullet points or a table-like format to clearly display details (such as name, location, operating hours, etc.).
                      * Do not omit any records.
                    - **Vector Database Results:**
                      * Summarize the key findings while referencing any notable details.
                      * Ensure the summary remains relevant to the user's query.
                2. If no tool output is available, generate your answer solely based on the user query, intent, and conversation history.
                3. For follow-up queries, incorporate previous context from the conversation history to maintain continuity.
                4. If the intent is 'out_of_scope', respond politely that you can only answer questions related to malls, stores, offers, events, or services.
                5. Keep your response concise yet complete—offer enough detail to be helpful without overwhelming the user.
                6. Add a friendly closing, such as asking if the user needs more information or has another question.

                Context Information:
                --- START CONTEXT ---
                User Query: {user_query}
                Intent: {intent}
                Conversation History: {conversation_history}
                Tool Outputs: {context_information}
                --- END CONTEXT ---

                Generate a response that follows these guidelines.
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
            "conversation_history": conversation_history_str,
            "context_information": context_str
        })
        updated_state = state.copy()
        updated_state["response"] = llm_response_text
        updated_state["next_node"] = "memory_node"  # Adjusted to route to memory_node
        print(f"LLM Response: {llm_response_text}")
        return updated_state

    except Exception as e:
        error_message = f"Error generating LLM response: {e}"
        print(error_message)
        updated_state = state.copy()
        updated_state["response"] = "Sorry, I am having trouble understanding you right now. Please try again later."
        updated_state["next_node"] = "memory_node"  # Adjusted to route to memory_node
        return updated_state