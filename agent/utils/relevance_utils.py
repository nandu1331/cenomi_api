from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config

config = load_config()

def evaluate_relevance_function(user_query: str, tool_output: str, intent) -> float:
    llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key)
    output_parser = StrOutputParser()

    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system",
            """
            You are an expert at evaluating relevance of tool output to a user query for Cenomi Malls chatbot.
            Assess how well the Tool Output answers the User Query.
            
            User Query: {user_query}
            Tool Output: {tool_output}
            Intent: {intent}
            
            Return a relevance score (0.0 to 1.0) based on relevance, accuracy, and completeness.
            Return only the score as a decimal number.
            """),
        ("human", "{user_query}\n{tool_output}")
    ])

    try:
        relevance_chain = relevance_prompt | llm | output_parser
        return float(relevance_chain.invoke({"user_query": user_query, "tool_output": tool_output, "intent": intent}).strip())
    except Exception:
        return 0.0