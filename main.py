# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.agent_graph import create_agent_graph
from typing import Optional


app = FastAPI(title="Cenomi AI Chatbot API")


class ChatRequest(BaseModel):
    text: str
    user_id: str
    language: str


class ChatResponse(BaseModel):
    message: str


agent_graph = create_agent_graph()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to handle chat requests from the frontend.
    """
    try:
        
        input_data = {
            "user_query": request.text,
            "conversation_history": [],
            "user_id": request.user_id,
            "language": request.language
        }

        result = agent_graph.invoke(input_data, {"recursion_limit": 500})

        response_text = result.get("response", "Sorry, I couldn't process your request.")
        return ChatResponse(message=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)