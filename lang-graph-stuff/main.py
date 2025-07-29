# # main.py
# """
# FastAPI app for SaaS optimization backend.
# POST /search: {query: str, session_id: str} -> {use_case, case_study, insights, message}
# """
# import os
# from fastapi import FastAPI, Request, HTTPException
# from pydantic import BaseModel
# from agent import run_agent
# from dotenv import load_dotenv
# from loguru import logger

# load_dotenv()

# app = FastAPI(title="SaaS Optimization Backend")

# class SearchRequest(BaseModel):
#     query: str
#     session_id: str

# class SearchResponse(BaseModel):
#     use_case: str
#     case_study: str
#     insights: list[str]
#     message: str

# @app.post("/search", response_model=SearchResponse)
# async def search(req: SearchRequest):
#     try:
#         logger.info(f"Received query: {req.query} (session: {req.session_id})")
#         result = run_agent(req.query, req.session_id)
#         return SearchResponse(**result)
#     except Exception as e:
#         logger.error(f"Error: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# # Sample queries for testing:
# # 'increase users' should return onboarding-related results
# # 'payment fraud' should acknowledge no match


# main.py
"""
FastAPI app for SaaS optimization backend.
POST /search: {query: str, session_id: str} -> {use_case, case_study, insights, message}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_agent
from dotenv import load_dotenv
from loguru import logger  # Ensure installed: pip install loguru

load_dotenv()

app = FastAPI(title="SaaS Optimization Backend")

class SearchRequest(BaseModel):
    query: str
    session_id: str

class SearchResponse(BaseModel):
    use_case: str
    case_study: str
    insights: list[str]
    message: str

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        logger.info(f"Received query: {req.query} (session: {req.session_id})")
        result = run_agent(req.query, req.session_id)
        if not result:
            return SearchResponse(use_case="", case_study="", insights=[], message="No results found.")
        return SearchResponse(**result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Sample queries for testing:
# 'increase users' should return onboarding-related results
# 'payment fraud' should acknowledge no match
