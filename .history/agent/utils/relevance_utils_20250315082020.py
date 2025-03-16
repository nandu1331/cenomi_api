from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.config_loader import load_config
from nodes.intent_router_node import IntentCategory

config = load_config()

# --- (Assume config is loaded in tool_selection_node.py or import it here) ---

def evaluate_relevance_function(user_query: str, tool_output: str, intent: IntentCategory) -> float:
    """
    Evaluates the relevance of a tool's output to the user query using an LLM.
    Returns a relevance score (0.0 to 1.0), higher is more relevant.
    """
    print("--- evaluate_relevance_function ---")
    print(f"Evaluating relevance for User Query: {user_query}")
    print(f"Tool Output to Evaluate:\n{tool_output}")

    llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")
    output_parser = StrOutputParser()

    relevance_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
                You are an expert at evaluating the relevance of a tool's output to a user's query in a chatbot system for Cenomi Malls.
                Your task is to assess how well the provided 'Tool Output' answers the 'User Query'.

                User Query: {user_query}
                Tool Output:
                --- START TOOL OUTPUT ---
                {tool_output}
                --- END TOOL OUTPUT ---

                Intent Category (for context, may be helpful): {intent}

                Evaluation Criteria:
                - Relevance: How directly and effectively does the Tool Output address the User Query?
                - Accuracy: Is the information in the Tool Output accurate and factual based on common sense? (No need to verify against external sources, just basic plausibility).
                - Completeness (to a lesser extent): Does the Tool Output provide a reasonably complete answer or at least a good starting point for an answer?

                Output:
                Return a relevance score as a decimal number between 0.0 and 1.0, where:
                - 1.0 = Highly relevant, the Tool Output directly and accurately answers the User Query.
                - 0.7 - 0.9 = Moderately relevant, the Tool Output is somewhat relevant and provides some useful information, but might not fully answer the query or might have minor issues.
                - 0.4 - 0.6 = Slightly relevant, the Tool Output has a weak connection to the User Query, might be tangentially related or only partially useful.
                - Below 0.4 = Not relevant, the Tool Output does not address the User Query or is completely irrelevant.

                Return ONLY the relevance score as a plain decimal number (e.g., '0.85', '0.2', '1.0'). Do not include any explanations or comments.

                Relevance Score:
                """),
            ("human", "User Query: {user_query}\nTool Output: {tool_output}"), # Redundant human message, could remove
        ]
    )

    relevance_chain = relevance_prompt | llm | output_parser

    try:
        relevance_score_str = relevance_chain.invoke({
            "user_query": user_query,
            "tool_output": tool_output,
            "intent": intent, # Pass intent as context
        })
        relevance_score = float(relevance_score_str.strip()) # Convert string to float and strip whitespace
        print(f"Raw Relevance Score from LLM: {relevance_score_str}, Parsed Score: {relevance_score}")
        return relevance_score

    except Exception as e:
        error_message = f"Error evaluating relevance: {e}"
        print(error_message)
        return 0.0 # Default to 0.0 relevance in case of error