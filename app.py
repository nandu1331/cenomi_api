from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agent.agent_graph import create_agent_graph
from agent.utils.database_utils import db_fetch_all
from uuid import uuid4

app = FastAPI()

# Initialize the agent graph
graph = create_agent_graph()

# Request models
class ChatRequest(BaseModel):
    user_query: str
    role: str
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    language: str
    mall_id: int
    mall_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class UpdateRequest(BaseModel):
    user_query: str
    user_id: int

# Mock tenant database (replace with real DB in production)
tenants = {
    "tenant@example.com": {"password": "password123", "user_id": 1}
}

@app.post("/login")
async def login(request: LoginRequest):
    """Tenant login endpoint."""
    tenant = tenants.get(request.email)
    if tenant and tenant["password"] == request.password:
        return {"user_id": tenant["user_id"], "role": "TENANT"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/malls")
async def get_malls():
    malls = db_fetch_all("SELECT mall_id, name_en FROM malls")
    return [{"mall_id": str(mall["mall_id"]), "name_en": mall["name_en"]} for mall in malls]

@app.post("/chat")
async def chat(request: ChatRequest):
    """General chat endpoint for customer or tenant queries."""
    
    session_id = request.session_id if request.session_id else str(uuid4())
    
    input_data = {
        "user_query": request.user_query,
        "role": request.role,
        "user_id": request.user_id,
        "language": request.language,
        "session_id": request.session_id,
        "mall_id": request.mall_id,
        "mall_name": request.mall_name
    }
    session_id = request.session_id
    config = {"configurable": {"thread_id": session_id}}
    try:
        # Stream the graph and collect the final state
        final_state = None
        for event in graph.stream(input_data, config=config):
            final_state = event
        # Extract response from the final state
        response = final_state.get("output_node", {}).get("response", "No response generated.")
        return {"response": response, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    """Tenant update endpoint for CRUD operations."""
    input_data = {
        "user_query": request.user_query,
        "role": "TENANT",
        "user_id": request.user_id
    }
    try:
        # Stream the graph and collect the final state
        final_state = None
        for event in graph.stream(input_data, {"recursion_limit": 500}):
            final_state = event
        # Extract response from the final state
        response = final_state.get("output_node", {}).get("response", "No response generated.")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing update request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)