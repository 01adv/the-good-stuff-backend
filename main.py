
# main.py
"""
FastAPI app for SaaS optimization backend.
POST /search: {query: str, session_id: str} -> {use_case, case_study, insights, message}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lang_graph_stuff.agent import run_agent
from dotenv import load_dotenv
from loguru import logger  # Ensure installed: pip install loguru
from typing import List



load_dotenv()


app = FastAPI(title="SaaS Optimization Backend")

# Recursively convert custom objects to dicts/lists/strings
def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
        return to_serializable(obj.dict())
    elif hasattr(obj, '__dict__'):
        return to_serializable(vars(obj))
    elif hasattr(obj, '__str__') and not isinstance(obj, (str, bytes)):
        return str(obj)
    else:
        return obj

class SearchRequest(BaseModel):
    query: str
    session_id: str

# class SearchResponse(BaseModel):
#     use_case: str
#     case_study: str
#     insights: list[str]
#     message: str

class SearchItem(BaseModel):
    title: str
    url: str
    category: str

class SearchResponse(BaseModel):
    use_case: List[SearchItem]
    case_study: List[SearchItem] 
    insights: List[SearchItem]
    message: str


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        logger.info(f"Received query: {req.query} (session: {req.session_id})")
        result = run_agent(req.query, req.session_id)
        if not result:
            return SearchResponse(use_case=[], case_study=[], insights=[], message="No results found.")
        # Convert any custom objects to serializable types
        result = to_serializable(result)
        return SearchResponse(**result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Sample queries for testing:
# 'increase users' should return onboarding-related results
# 'payment fraud' should acknowledge no match
