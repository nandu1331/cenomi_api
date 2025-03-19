from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from config.config_loader import load_config

config = load_config()
llm = ChatGoogleGenerativeAI(model=config.llm.model_name, api_key=config.llm.api_key)
json_output_parser = JsonOutputParser()

field_extraction_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
        """
        You are an expert at extracting structured information from user queries related to tenant data operations in a mall database.
        Extract relevant fields and their values from the user's query for the specified entity type.
        
        Entity Type: {entity_type}
        User Query: {user_query}
        
        Return a valid JSON object with field-value pairs. Include only fields with reliably extracted values.
        Return an empty JSON object ({}) if no fields can be extracted.
        """),
    ("human", "Extract fields and values from the user query for {entity_type} as JSON only.")
])

def extract_fields_from_query(user_query: str, entity_type: str) -> Dict[str, Any]:
    try:
        field_extraction_chain = field_extraction_prompt_template | llm | json_output_parser
        return field_extraction_chain.invoke({"user_query": user_query, "entity_type": entity_type})
    except Exception:
        return {}