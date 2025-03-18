from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.agent_graph import create_agent_graph
from agent.utils.database_utils import get_db_connection  # Assuming this exists

app = FastAPI(title="Cenomi AI Chatbot API")

class ChatRequest(BaseModel):
    text: str
    user_id: str
    language: str

class ChatResponse(BaseModel):
    message: str

agent_graph = create_agent_graph()

def fetch_conversation_history(user_id: str) -> list[dict[str, str]]:
    """Fetch conversation history from the chat_history table for a given user."""
    connection = get_db_connection()
    if not connection:
        return []
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT user_query, bot_response 
                FROM chat_history 
                WHERE user_id = %s 
                ORDER BY timestamp ASC
            """, (user_id,))
            rows = cursor.fetchall()
            return [{"user": row[0], "bot": row[1]} for row in rows]
    except Exception as e:
        print(f"Error fetching conversation history: {e}")
        return []
    finally:
        connection.close()

def store_conversation_history(user_id: str, user_query: str, bot_response: str):
    """Store a new interaction in the chat_history table."""
    connection = get_db_connection()
    if not connection:
        return
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO chat_history (user_id, user_query, bot_response)
                VALUES (%s, %s, %s)
            """, (user_id, user_query, bot_response))
            connection.commit()
    except Exception as e:
        print(f"Error storing conversation history: {e}")
    finally:
        connection.close()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to handle chat requests from the frontend.
    """
    try:
        user_id = request.user_id
        # Fetch existing conversation history from the database
        conversation_history = fetch_conversation_history(user_id)

        input_data = {
            "user_query": request.text,
            "conversation_history": conversation_history,
            "user_id": user_id,
            "language": request.language
        }
        result = agent_graph.invoke(input_data, {"recursion_limit": 500})
        response_text = result.get("response", "Sorry, I couldn't process your request.")
        
        # Store the new interaction in the database
        store_conversation_history(user_id, request.text, response_text)
        
        return ChatResponse(message=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)