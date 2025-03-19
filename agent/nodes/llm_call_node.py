from agent.agent_state import AgentState
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_loader import load_config

config = load_config()
llm = ChatGoogleGenerativeAI(model=config.llm.model_name, google_api_key=config.llm.api_key)
output_parser = StrOutputParser()

def llm_call_node(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    intent = state["intent"]
    tool_output = state.get("tool_outputs", {})
    
    conversation_history_list = state.get("conversation_history", [])
    conversation_history_str = "\n".join(
        [f"User: {turn['user']}\nAssistant: {turn['bot']}" 
         for turn in conversation_history_list if 'user' in turn and 'bot' in turn]
    ) or "No previous conversation history available."

    context_str = "Tool Output:\n" + (
        "\n".join(f"{tool_name}:\n{output}" for tool_name, output in tool_output.items())
        if tool_output else "No tool output generated.\n"
    )

    response_prompt = ChatPromptTemplate.from_messages([
        ("system",
            """
            You are Cenomi Chatbot, a funny and engaging chatbot assistant for Cenomi Malls.
            Your goal is to provide helpful, informative, and entertaining responses to users.
            Use the provided context—including the user query, intent, conversation history, and any tool outputs—to craft your answer.

            Guidelines:
            1. If tool output is provided:
               - Structured SQL Results: Present the complete list of records using bullet points or a table-like format.
               - Vector Database Results: Summarize key findings while referencing notable details.
            2. If no tool output, base your answer on the user query, intent, and conversation history.
            3. For follow-ups, use conversation history for continuity.
            4. For 'out_of_scope', say you can only answer mall-related questions.
            5. Keep responses concise yet complete.
            6. End with a friendly closing.

            Context Information:
            User Query: {user_query}
            Intent: {intent}
            Conversation History: {conversation_history}
            Tool Outputs: {context_information}
            """),
        ("human", "{user_query}"),
    ])

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
        updated_state["next_node"] = "memory_node"
        return updated_state
    except Exception as e:
        updated_state = state.copy()
        updated_state["response"] = "Sorry, I am having trouble understanding you right now. Please try again later."
        updated_state["next_node"] = "memory_node"
        return updated_state