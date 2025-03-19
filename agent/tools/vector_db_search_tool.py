from typing import Any
from langchain.tools import BaseTool
from agent.utils.database_utils import get_vector_db_client, get_vector_db_collection
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDBSearchTool(BaseTool):
    name: str = "vector_database_search"
    description: str = "Useful for searching relevant information in the mall database based on semantic similarity. Input should be a natural language query."

    def _run(self, query: str, context: str = None) -> str:
        chroma_client = get_vector_db_client()
        chroma_collection = get_vector_db_collection(chroma_client)
        
        if not chroma_collection:
            return "Error: Could not access VectorDB collection."
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        combined_query = f"{context} {query}" if context else query
        query_embedding = embedding_model.embed_query(combined_query)
        
        try:
            results = chroma_collection.query(query_embeddings=[query_embedding], n_results=5)
            if not results or not results['ids'] or not results['ids'][0] or not results['documents'] or not results['documents'][0]:
                return "No relevant information found in the mall database for your query."
            
            output = "Vector Database Search Results:\n\n"
            for i, document in enumerate(results['documents'][0]):
                output += f"--- Result {i+1} ---\nDocument Snippet: {document[:200]}...\n"
                if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] and results['metadatas'][0][i]:
                    output += f"Metadata: {results['metadatas'][0][i]}\n"
                output += "\n"
            return output
        except Exception as e:
            return f"Error during VectorDB search: {e}"
        
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Asynchronous _arun method not implemented for VectorDBSearchTool.")