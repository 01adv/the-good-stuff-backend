
# agent.py
"""
LangGraph agent for orchestrating query processing.
Nodes: Input, Routing, Retrieval, Generation, Output.
Session memory via MemorySaver with conversation history.
"""
import os
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from db import pinecone_db
from dotenv import load_dotenv
import time
from pydantic import BaseModel, Field

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("agent")

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"

# Session memory config
MEMORY_LIMIT = 3  # Last 3 interactions

saver = MemorySaver()

class AgentState(BaseModel):
    query: str
    session_id: str
    conversation_history: List[Dict] = Field(default_factory=list)  # Current conversation context
    embedding: List[float] = None
    buckets: List[str] = Field(default_factory=list)
    funnel_stage: str = None
    retrieval_results: Dict[str, List] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)

# Structured output models
class RoutingOutput(BaseModel):
    buckets: List[str]
    funnel_stage: str = None

class SearchItem(BaseModel):
    title: str = Field(description="The title of the source document")
    url: str = Field(description="The URL of the source document") 
    category: str = Field(description="The specific category or subcategory this item belongs to")

class GenerationOutput(BaseModel):
    use_case: List[SearchItem] = Field(default_factory=list, description="List of use case items with title, URL and category")
    case_study: List[SearchItem] = Field(default_factory=list, description="List of case study items with title, URL and category")
    insights: List[SearchItem] = Field(default_factory=list, description="List of insight items with title, URL and category")
    message: str = Field(description="Overall message or summary")

# Helper function to convert ScoredVector to serializable dict
def scored_vector_to_dict(scored_vector):
    """Convert Pinecone ScoredVector to serializable dictionary"""
    return {
        "id": scored_vector.id,
        "score": float(scored_vector.score),
        "values": list(scored_vector.values) if scored_vector.values else [],
        "metadata": dict(scored_vector.metadata) if scored_vector.metadata else {}
    }

# Prompts (enhanced with memory)
ROUTING_PROMPT = """Analyze query: {query} with past conversation context: {conversation_history}. 
Route to relevant buckets from: services, case-studies, insights. 
Map to funnel stages (awareness, consideration, decision) if similarity > 0.4. 
Output structured JSON with buckets and funnel_stage."""

GENERATION_PROMPT = """Given context: {context}, funnel_stage: {funnel_stage}, query: {query}, and past conversation: {conversation_history}, generate structured JSON with:
- use_case: array of objects with title, url, category
- case_study: array of objects with title, url, category  
- insights: array of objects with title, url, category
- message: string summary
- Avoid using curly or typographic quotes; use straight quotes (') only.

Extract title, url, and category from the context metadata. If no confident match (similarity < 0.4), return empty arrays and acknowledge in message with suggestions for related services.
Consider the conversation history to provide more personalized and contextual responses."""

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

routing_llm = llm.with_structured_output(RoutingOutput)
generation_llm = llm.with_structured_output(GenerationOutput)

# --- Node functions ---
def input_node(state: AgentState) -> AgentState:
    logger.info(f"Input node: query='{state.query}', session_id='{state.session_id}'")
    state.embedding = embeddings.embed_query(state.query)
    logger.info(f"Generated embedding: {state.embedding[:5]}...")  # Show only first 5 values
    return state

def routing_node(state: AgentState) -> AgentState:
    # Use conversation history for context
    history_str = str(state.conversation_history[-MEMORY_LIMIT:]) if state.conversation_history else ""
    logger.info(f"Routing node: query='{state.query}', conversation_history='{history_str}'")
    
    resp = routing_llm.invoke(ROUTING_PROMPT.format(
        query=state.query, 
        conversation_history=history_str
    ))
    
    logger.info(f"Routing LLM output: buckets={resp.buckets}, funnel_stage={resp.funnel_stage}")
    state.buckets = resp.buckets
    state.funnel_stage = resp.funnel_stage
    return state

def retrieval_node(state: AgentState) -> AgentState:
    logger.info(f"Retrieval node: buckets={state.buckets}")
    if not state.buckets:
        logger.warning("No buckets found for retrieval.")
        return state  # Fallback: no retrieval
    
    for bucket in state.buckets:
        res = pinecone_db.query(bucket, state.embedding, top_k=3)
        logger.info(f"Retrieved for bucket '{bucket}': {len(res['matches'])} matches")
        
        # Convert ScoredVector objects to serializable dictionaries
        serializable_matches = []
        for match in res["matches"]:
            serializable_matches.append(scored_vector_to_dict(match))
        
        state.retrieval_results[bucket] = serializable_matches
    
    return state

def generation_node(state: AgentState) -> AgentState:
    # Check confidence (filter low-similarity results)
    context = {}
    for bucket, matches in state.retrieval_results.items():
        high_conf = [m for m in matches if m["score"] > 0.4]
        context[bucket] = high_conf or []
    
    context_str = str(context)
    history_str = str(state.conversation_history[-MEMORY_LIMIT:]) if state.conversation_history else ""
    
    logger.info(f"Generation node: query={state.query}, funnel_stage={state.funnel_stage}")
    
    resp = generation_llm.invoke(GENERATION_PROMPT.format(
        context=context_str, 
        funnel_stage=state.funnel_stage, 
        query=state.query, 
        conversation_history=history_str
    ))
    
    logger.info(f"Generation LLM output: {resp.dict()}")
    state.output = resp.dict()

    # If no confident matches, keep LLM's message but ensure arrays are empty
    if not any(context.values()):  # No confident matches
        logger.warning("No confident matches found in context.")
        state.output["use_case"] = []
        state.output["case_study"] = []
        state.output["insights"] = []
    
    return state

def output_node(state: AgentState) -> AgentState:
    logger.info(f"Output node: output={state.output}")
    
    # Update conversation history with this interaction
    interaction = {
        "query": state.query,
        "response": state.output,
        "timestamp": time.time(),
        "funnel_stage": state.funnel_stage,
        "buckets": state.buckets
    }
    
    # Add to conversation history and maintain limit
    state.conversation_history.append(interaction)
    if len(state.conversation_history) > MEMORY_LIMIT:
        state.conversation_history = state.conversation_history[-MEMORY_LIMIT:]
    
    logger.info(f"Updated conversation history: {len(state.conversation_history)} interactions")
    return state

# --- Graph definition ---
graph = StateGraph(AgentState)
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

app = graph.compile(checkpointer=saver)  # Compile with checkpointer

def run_agent(query: str, session_id: str) -> Dict[str, Any]:
    logger.info(f"Running agent for session_id='{session_id}' with query='{query}'")
    config = {"configurable": {"thread_id": session_id}}  # For session threading
    
    try:
        # Get previous state from checkpointer to maintain conversation history
        state = AgentState(query=query, session_id=session_id)
        
        result = app.invoke(state, config=config)
        logger.info(f"Agent completed successfully")
        return result['output']

    except Exception as e:
        logger.exception(f"Agent run failed for session_id='{session_id}' with query='{query}'. Error: {str(e)}")

        return {
            "use_case": [],
            "case_study": [],
            "insights": [],
            "message": "Something went wrong while processing your query. Please try again later or contact support."
        }