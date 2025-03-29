from typing import Any
from langchain.tools import BaseTool
from agent.utils.database_utils import get_vector_db_client, get_vector_db_index
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDBSearchTool(BaseTool):
    """Tool to perform vector database search and retrieve relevant data using Pinecone."""
    
    name: str = "vector_database_search"
    description: str = (
        "Useful for searching relevant information in the mall database based on semantic similarity. "
        "Input should be a natural language query to search for relevant documents."
    )
    
    def _run(self, query: str, context: str = None, mall_name: str = None) -> str:
        """Use the Pinecone vector database to search for semantically similar documents."""
        
        # Initialize Pinecone client and index
        vector_db_client = get_vector_db_client()
        index_name = "mall-index"  # Match the index used in ingestion
        vector_db_index = get_vector_db_index(vector_db_client, index_name)
        
        if not vector_db_index:
            return "Error: Could not access Pinecone index."
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        
        # Combine context and query if context is provided
        combined_query = f"{context} {query}" if context else query
        query_embedding = embedding_model.embed_query(combined_query)
        
        try:
            # Query Pinecone
            results = vector_db_index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True  # Retrieve metadata stored in Pinecone
            )
            
            if not results or not results.get("matches"):
                return "No relevant information found in the mall database for your query."
            
            # Format results
            output_string = "Vector Database Search Results:\n\n"
            for i, match in enumerate(results["matches"]):
                output_string += f"--- Result {i+1} ---\n"
                metadata = match["metadata"]
                document = metadata.get("document", "No document text available")
                output_string += f"Document Snippet: {document[:200]}...\n"
                output_string += f"Metadata: {metadata}\n\n"
                
            return output_string
        
        except Exception as e:
            return f"Error during VectorDB search: {e}"
        
    async def _arun(self, query: str) -> str:
        """Asynchronous run method (not implemented for this basic tool)."""
        raise NotImplementedError("Asynchronous _arun method not implemented for VectorDBSearchTool.")