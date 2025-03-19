from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.agent_graph import create_agent_graph
from agent.utils.database_utils import get_db_connection

app = FastAPI(title="Cenomi AI Chatbot API")

class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    text: str
    user_id: str
    language: str

class ChatResponse(BaseModel):
    message: str

agent_graph = create_agent_graph()

def fetch_user_details(user_id: str) -> dict:
    connection = get_db_connection()
    if not connection:
        return {}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT role, store_id FROM users WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            return {"role": result[0], "store_id": result[1]} if result else {}
    finally:
        connection.close()

def fetch_conversation_history(user_id: str) -> list[dict[str, str]]:
    connection = get_db_connection()
    if not connection:
        return []
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_query, bot_response FROM chat_history WHERE user_id = %s ORDER BY timestamp ASC", (user_id,))
            rows = cursor.fetchall()
            return [{"user": row[0], "bot": row[1]} for row in rows]
    finally:
        connection.close()

def store_conversation_history(user_id: str, user_query: str, bot_response: str):
    connection = get_db_connection()
    if not connection:
        return
    try:
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO chat_history (user_id, user_query, bot_response) VALUES (%s, %s, %s)", (user_id, user_query, bot_response))
            connection.commit()
    finally:
        connection.close()

@app.post("/login")
async def login(request: LoginRequest):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection error")
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id, role, store_id FROM users WHERE email = %s AND password = %s", (request.email, request.password))
            user = cursor.fetchone()
            if user:
                return {"user_id": str(user[0]), "role": user[1], "store_id": user[2]}
            raise HTTPException(status_code=401, detail="Invalid credentials")
    finally:
        connection.close()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_id = request.user_id
        user_details = fetch_user_details(user_id)
        conversation_history = fetch_conversation_history(user_id)

        input_data = {
            "user_query": request.text,
            "conversation_history": conversation_history,
            "user_id": user_id,
            "language": request.language,
            "role": user_details.get("role", "anonymous"),
            "store_id": user_details.get("store_id")
        }
        result = agent_graph.invoke(input_data, {"recursion_limit": 500})
        response_text = result.get("response", "Sorry, I couldn't process your request.")
        
        store_conversation_history(user_id, request.text, response_text)
        return ChatResponse(message=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)