from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from typing import List
from config.config_loader import load_config
import ast

config = load_config()

required_fields_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
        """
        You are an expert at determining required fields for tenant data operations in a mall database.
        Analyze the intent, query, schema, and existing fields to identify essential fields for the operation.
        
        Database Schema: {db_schema}
        Tenant Intent: {intent}
        User Query: {user_query}
        Existing fields: {existing_fields}
        
        For 'insert': Include all NOT NULL fields not auto-generated.
        For 'update': Include fields to identify the record plus one to update.
        For 'delete': Include fields to identify the record.
        Return a Python list of strings with required field names not in existing fields.
        """),
    ("human", "Determine the required fields for {intent} based on the user query, schema, and existing fields.")
])

class RequiredFieldsCalculator:
    def __init__(self, model_name=config.llm.model_name, api_key=config.llm.api_key, cache_ttl=3600):
        self.llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
        self.output_parser = StrOutputParser()
        self.cache = {}
        self.cache_ttl = cache_ttl
        
    def calculate_required_fields(self, intent, user_query: str, db_schema: str, existing_fields: str) -> List[str]:
        cache_key = f"{intent.value}|{user_query}|{hash(db_schema)}"
        if cache_key in self.cache and (time.time() - self.cache[cache_key].get('timestamp', 0)) < self.cache_ttl:
            return self.cache[cache_key]['fields']
        
        try:
            required_fields_chain = required_fields_prompt_template | self.llm | self.output_parser
            response_str = required_fields_chain.invoke({
                "intent": intent.value, "user_query": user_query, "db_schema": db_schema, "existing_fields": existing_fields
            })
            fields = ast.literal_eval(response_str.strip())
            if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
                return []
            
            validated_fields = [f.strip() for f in fields if f.strip() in db_schema]
            self.cache[cache_key] = {'fields': validated_fields, 'timestamp': time.time()}
            return validated_fields
        except Exception:
            return []