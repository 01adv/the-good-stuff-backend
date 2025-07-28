# agent.py
"""
LangGraph agent for orchestrating query processing.
Nodes: Input, Routing, Retrieval, Generation, Output.
Session memory via CheckpointSaver (SqliteSaver for dev).
"""
import os
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import SqliteSaver
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from db import pinecone_db
from dotenv import load_dotenv
import time

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"

# Session memory
MEMORY_PATH = "session_memory.sqlite"
MEMORY_EXPIRY = 600  # 10 minutes
MEMORY_LIMIT = 3

saver = SqliteSaver(MEMORY_PATH)

# Prompts
ROUTING_PROMPT = """Analyze query: {query}. Route to buckets: services, case-studies, insights. Map to funnel stages if similarity > 0.8. Return JSON: {"buckets": [..], "funnel_stage": "..."}"""
GENERATION_PROMPT = """Given context from buckets: {context}, funnel_stage: {funnel_stage}, and query: {query}, generate a JSON with use_case, case_study, insights (3), and message. If no confident match, acknowledge and suggest related services."""

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# --- Node functions ---
def input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    state["embedding"] = embeddings.embed_query(query)
    return state

def routing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    # Use LLM to decide buckets and funnel stage
    resp = llm.invoke(ROUTING_PROMPT.format(query=query))
    try:
        routing = resp.additional_kwargs.get("function_call", {}).get("arguments")
        if routing:
            import json
            routing = json.loads(routing)
        else:
            routing = {"buckets": [], "funnel_stage": None}
    except Exception:
        routing = {"buckets": [], "funnel_stage": None}
    state["buckets"] = routing["buckets"]
    state["funnel_stage"] = routing["funnel_stage"]
    return state

def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    embedding = state["embedding"]
    buckets = state.get("buckets", [])
    results = {}
    for bucket in buckets:
        res = pinecone_db.query(bucket, embedding, top_k=3)
        results[bucket] = res["matches"]
    state["retrieval_results"] = results
    return state

def generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    context = state.get("retrieval_results", {})
    funnel_stage = state.get("funnel_stage")
    query = state["query"]
    # Compose context for LLM
    context_str = str(context)
    resp = llm.invoke(GENERATION_PROMPT.format(context=context_str, funnel_stage=funnel_stage, query=query))
    try:
        import json
        output = json.loads(resp.content)
    except Exception:
        output = {"use_case": "", "case_study": "", "insights": [], "message": "No direct match, but explore our related onboarding services."}
    state["output"] = output
    return state

def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return state["output"]

# --- Graph definition ---
graph = StateGraph()
graph.add_node("input", input_node)
graph.add_node("routing", routing_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("generation", generation_node)
graph.add_node("output", output_node)

graph.add_edge("input", "routing")
graph.add_edge("routing", "retrieval")
graph.add_edge("retrieval", "generation")
graph.add_edge("generation", "output")
graph.add_edge("output", END)

graph.set_entry_point("input")
graph.set_exit_point("output")

def run_agent(query: str, session_id: str) -> Dict[str, Any]:
    # Restore session memory (last 3 interactions, expire after 10 min)
    now = int(time.time())
    memory = saver.load(session_id)
    if memory:
        memory = [m for m in memory if now - m["timestamp"] < MEMORY_EXPIRY]
        memory = memory[-MEMORY_LIMIT:]
    else:
        memory = []
    state = {"query": query, "session_id": session_id, "memory": memory}
    result = graph.run(state)
    # Save new memory
    memory.append({"query": query, "timestamp": now, "result": result})
    saver.save(session_id, memory)
    return result
