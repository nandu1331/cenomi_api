from typing import Any, Dict, List
from langchain.tools import BaseTool
from utils.database_utils import get_vector_db_client, get_vector_db_collection
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDBSearchTool(BaseTool):
    """Tool to perform vector database search and retrieve relevant data."""
    
    name: str = "vector_database_search"
    description: str = (
        "Useful for searching relevant information in the mall database based on semantic similarity. "
        "Input should be a natural language query to search for relevant documents."
    )
    
    def _run(self, query: str) -> str:
        """Use the ChromaDB vector database to search for semantically similar documents."""
        print(f"--- VectorDBSearchTool: Running with query: {query} ---")
        
        chroma_client = get_vector_db_client()
        chroma_collection = get_vector_db_collection(chroma_client)
        
        if not chroma_collection:
            return "Error: Could not access VectorDB collection."
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        query_embedding = embedding_model.embed_query(query)
        
        try:
            results = chroma_collection.query(
                query_embeddings=[[query_embedding]],
                n_results=3
            )
            
            if not results or not results['ids'] or not results['ids'][0] or not results['documents'] or not results['documents'][0]: 
                return "No relevant information found in the mall database for your query."
            
            output_string = "Vector Database Search Results:\n\n"
            for i, document in enumerate(results['documents'][0]):
                output_string += f"--- Result {i+1} ---\n"
                output_string += f"Document Snippet: {document[:200]}...\n"
                if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] and results['metadatas'][0][i]:
                    metadata = results['metadatas'][0][i]
                    output_string += f"Metadata: {metadata}\n"
                output_string += "\n"
                
            print("VectorDBSearchTool Results:\n", output_string)
            return output_string
        
        except Exception as e:
            error_message = f"Error during VectorDB search: {e}"
            print(error_message)
            return error_message
        
    async def _arun(self, query: str) -> str:
        """Asynchronous run method (not implemented for this basic tool)."""
        raise NotImplementedError("Asynchronous _arun method not implemented for VectorDBSearchTool.")