# # agent.py
# """
# LangGraph agent for orchestrating query processing.
# Nodes: Input, Routing, Retrieval, Generation, Output.
# Session memory via CheckpointSaver (SqliteSaver for dev).
# """
# import os
# from typing import Dict, Any, List
# from langgraph.graph import StateGraph, END
# # from langgraph.checkpoint import SqliteSaver
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from db import pinecone_db
# from dotenv import load_dotenv
# import time

# from pydantic import BaseModel


# load_dotenv()

# EMBEDDING_MODEL = "text-embedding-3-large"
# LLM_MODEL = "gpt-4o"

# # Session memory
# # MEMORY_PATH = "session_memory.sqlite"
# MEMORY_EXPIRY = 600  # 10 minutes
# MEMORY_LIMIT = 3

# # saver = SqliteSaver(MEMORY_PATH)
# # saver = MemorySaver(expiry=MEMORY_EXPIRY, limit=MEMORY_LIMIT)
# saver = MemorySaver()

# class AgentState(BaseModel):
#     query: str
#     session_id: str
#     memory: list = []
#     embedding: list = None
#     buckets: list = []
#     funnel_stage: str = None
#     retrieval_results: dict = {}
#     output: dict = {}



# # Prompts
# ROUTING_PROMPT = """Analyze query: {query}. Route to buckets: services, case-studies, insights. Map to funnel stages if similarity > 0.8. Return JSON: {"buckets": [..], "funnel_stage": "..."}"""
# GENERATION_PROMPT = """Given context from buckets: {context}, funnel_stage: {funnel_stage}, and query: {query}, generate a JSON with use_case, case_study, insights (3), and message. If no confident match, acknowledge and suggest related services."""

# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
# llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# # --- Node functions ---
# def input_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     query = state["query"]
#     state["embedding"] = embeddings.embed_query(query)
#     return state

# def routing_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     query = state["query"]
#     # Use LLM to decide buckets and funnel stage
#     resp = llm.invoke(ROUTING_PROMPT.format(query=query))
#     try:
#         routing = resp.additional_kwargs.get("function_call", {}).get("arguments")
#         if routing:
#             import json
#             routing = json.loads(routing)
#         else:
#             routing = {"buckets": [], "funnel_stage": None}
#     except Exception:
#         routing = {"buckets": [], "funnel_stage": None}
#     state["buckets"] = routing["buckets"]
#     state["funnel_stage"] = routing["funnel_stage"]
#     return state

# def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     embedding = state["embedding"]
#     buckets = state.get("buckets", [])
#     results = {}
#     for bucket in buckets:
#         res = pinecone_db.query(bucket, embedding, top_k=3)
#         results[bucket] = res["matches"]
#     state["retrieval_results"] = results
#     return state

# def generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     context = state.get("retrieval_results", {})
#     funnel_stage = state.get("funnel_stage")
#     query = state["query"]
#     # Compose context for LLM
#     context_str = str(context)
#     resp = llm.invoke(GENERATION_PROMPT.format(context=context_str, funnel_stage=funnel_stage, query=query))
#     try:
#         import json
#         output = json.loads(resp.content)
#     except Exception:
#         output = {"use_case": "", "case_study": "", "insights": [], "message": "No direct match, but explore our related onboarding services."}
#     state["output"] = output
#     return state

# def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     return state["output"]

# # --- Graph definition ---
# graph = StateGraph(AgentState)
# graph.add_node("input", input_node)
# graph.add_node("routing", routing_node)
# graph.add_node("retrieval", retrieval_node)
# graph.add_node("generation", generation_node)
# graph.add_node("output", output_node)

# graph.add_edge("input", "routing")
# graph.add_edge("routing", "retrieval")
# graph.add_edge("retrieval", "generation")
# graph.add_edge("generation", "output")
# graph.add_edge("output", END)

# graph.set_entry_point("input")
# graph.set_exit_point("output")

# def run_agent(query: str, session_id: str) -> Dict[str, Any]:
#     # Restore session memory (last 3 interactions, expire after 10 min)
#     now = int(time.time())
#     memory = saver.load(session_id)
#     if memory:
#         memory = [m for m in memory if now - m["timestamp"] < MEMORY_EXPIRY]
#         memory = memory[-MEMORY_LIMIT:]
#     else:
#         memory = []
#     state = {"query": query, "session_id": session_id, "memory": memory}
#     result = graph.run(state)
#     # Save new memory
#     memory.append({"query": query, "timestamp": now, "result": result})
#     saver.save(session_id, memory)
#     return result





# # agent.py
# """
# LangGraph agent for orchestrating query processing.
# Nodes: Input, Routing, Retrieval, Generation, Output.
# Session memory via SqliteSaver with expiry/limit.
# """
# import os
# import logging
# from typing import Dict, Any, List
# from langgraph.graph import StateGraph, END
# # from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from db import pinecone_db
# from dotenv import load_dotenv
# import time
# from pydantic import BaseModel, Field

# # --- Logging setup ---
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s %(message)s"
# )
# logger = logging.getLogger("agent")

# load_dotenv()

# EMBEDDING_MODEL = "text-embedding-3-large"
# LLM_MODEL = "gpt-4o"

# # Session memory config
# MEMORY_PATH = ":memory:"  # Use in-memory for dev; switch to file for persistence
# MEMORY_EXPIRY = 600  # 10 minutes
# MEMORY_LIMIT = 3  # Last 3 interactions

# # saver = SqliteSaver.from_conn_string(MEMORY_PATH)
# saver = MemorySaver()

# class AgentState(BaseModel):
#     query: str
#     session_id: str
#     memory: List[Dict] = Field(default_factory=list)  # Previous queries/results
#     embedding: List[float] = None
#     buckets: List[str] = Field(default_factory=list)
#     funnel_stage: str = None
#     retrieval_results: Dict[str, List] = Field(default_factory=dict)
#     output: Dict[str, Any] = Field(default_factory=dict)

# # Structured output models
# class RoutingOutput(BaseModel):
#     buckets: List[str]
#     funnel_stage: str = None

# # class GenerationOutput(BaseModel):
# #     use_case: str
# #     case_study: str
# #     insights: List[str]
# #     message: str

# class SearchItem(BaseModel):
#     title: str = Field(description="The title of the source document")
#     url: str = Field(description="The URL of the source document") 
#     category: str = Field(description="The specific category or subcategory this item belongs to")

# class GenerationOutput(BaseModel):
#     use_case: List[SearchItem] = Field(default_factory=list, description="List of use case items with title, URL and category")
#     case_study: List[SearchItem] = Field(default_factory=list, description="List of case study items with title, URL and category")
#     insights: List[SearchItem] = Field(default_factory=list, description="List of insight items with title, URL and category")
#     message: str = Field(description="Overall message or summary")


# # Prompts (enhanced with memory)
# ROUTING_PROMPT = """Analyze query: {query} with past context: {memory}. Route to buckets: services, case-studies, insights. Map to funnel stages if similarity > 0.4. Output structured JSON."""
# # GENERATION_PROMPT = """Given context: {context}, funnel_stage: {funnel_stage}, query: {query}, and past context: {memory}, generate structured JSON with use_case, case_study, insights (list of 3), and message. If no confident match (similarity < 0.4), acknowledge and suggest related services."""
# GENERATION_PROMPT = """Given context: {context}, funnel_stage: {funnel_stage}, query: {query}, and past context: {memory}, generate structured JSON with:
# - use_case: array of objects with title, url, category
# - case_study: array of objects with title, url, category  
# - insights: array of objects with title, url, category
# - message: string summary
# - Avoid using curly or typographic quotes; use straight quotes (') only.


# Extract title, url, and category from the context metadata. If no confident match (similarity < 0.4), return empty arrays and acknowledge in message with suggestions for related services."""


# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
# llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# routing_llm = llm.with_structured_output(RoutingOutput)
# generation_llm = llm.with_structured_output(GenerationOutput)

# # --- Node functions ---
# def input_node(state: AgentState) -> AgentState:
#     logger.info(f"Input node: query='{state.query}', session_id='{state.session_id}'")
#     state.embedding = embeddings.embed_query(state.query)
#     logger.info(f"Generated embedding: {state.embedding[:5]}...")  # Show only first 5 values
#     return state

# def routing_node(state: AgentState) -> AgentState:
#     memory_str = str(state.memory[-MEMORY_LIMIT:]) if state.memory else ""
#     logger.info(f"Routing node: query='{state.query}', memory='{memory_str}'")
#     resp = routing_llm.invoke(ROUTING_PROMPT.format(query=state.query, memory=memory_str))
#     logger.info(f"Routing LLM output####: {resp.dict()}")
#     logger.info(f"Routing LLM output: buckets={resp.buckets}, funnel_stage={resp.funnel_stage}")
#     state.buckets = resp.buckets
#     state.funnel_stage = resp.funnel_stage
#     return state

# def retrieval_node(state: AgentState) -> AgentState:
#     logger.info(f"Retrieval node: buckets={state.buckets}")
#     if not state.buckets:
#         logger.warning("No buckets found for retrieval.")
#         return state  # Fallback: no retrieval
#     for bucket in state.buckets:
#         res = pinecone_db.query(bucket, state.embedding, top_k=3)
#         logger.info(f"Retrieved ************ for bucket '{state.query}': '{bucket}': {res['matches']}")
#         state.retrieval_results[bucket] = res["matches"]
#     return state

# # def generation_node(state: AgentState) -> AgentState:
# #     # Check confidence (filter low-similarity results)
# #     context = {}
# #     for bucket, matches in state.retrieval_results.items():
# #         high_conf = [m for m in matches if m["score"] > 0.4]
# #         context[bucket] = high_conf or []  # Empty if no confident matches

# #     context_str = str(context)
# #     memory_str = str(state.memory[-MEMORY_LIMIT:]) if state.memory else ""
# #     logger.info(f"Generation node: context={context_str}, funnel_stage={state.funnel_stage}, query={state.query}")
# #     resp = generation_llm.invoke(GENERATION_PROMPT.format(context=context_str, funnel_stage=state.funnel_stage, query=state.query, memory=memory_str))
# #     logger.info(f"Generation LLM output: {resp.dict()}")
# #     state.output = resp.dict()
# #     if not any(context.values()):  # No confident matches
# #         logger.warning("No confident matches found in context.")
# #         state.output["message"] = "No direct match, but explore our related onboarding services."
# #     return state

# def generation_node(state: AgentState) -> AgentState:
#     # Check confidence (filter low-similarity results)
#     context = {}
#     for bucket, matches in state.retrieval_results.items():
#         high_conf = [m for m in matches if m["score"] > 0.4]
#         context[bucket] = high_conf or []
    
#     context_str = str(context)
#     memory_str = str(state.memory[-MEMORY_LIMIT:]) if state.memory else ""
    
#     logger.info(f"Generation node: context={context_str}, funnel_stage={state.funnel_stage}, query={state.query}")
    
#     resp = generation_llm.invoke(GENERATION_PROMPT.format(
#         context=context_str, 
#         funnel_stage=state.funnel_stage, 
#         query=state.query, 
#         memory=memory_str
#     ))
    
#     logger.info(f"Generation LLM output: {resp.dict()}")
#     state.output = resp.dict()
    
#     if not any(context.values()):  # No confident matches
#         logger.warning("No confident matches found in context.")
#         state.output["message"] = "No direct match, but explore our related onboarding services."
#         state.output["use_case"] = []
#         state.output["case_study"] = []  
#         state.output["insights"] = []
    
#     return state


# def output_node(state: AgentState) -> AgentState:
#     logger.info(f"Output node: output={state.output}")
#     return state

# # --- Graph definition ---
# graph = StateGraph(AgentState)
# graph.add_node("input", input_node)
# graph.add_node("routing", routing_node)
# graph.add_node("retrieval", retrieval_node)
# graph.add_node("generation", generation_node)
# graph.add_node("output", output_node)

# graph.add_edge("input", "routing")
# graph.add_edge("routing", "retrieval")
# graph.add_edge("retrieval", "generation")
# graph.add_edge("generation", "output")
# graph.add_edge("output", END)

# graph.set_entry_point("input")

# app = graph.compile(checkpointer=saver)  # Compile with checkpointer

# # def run_agent(query: str, session_id: str) -> Dict[str, Any]:
# #     logger.info(f"Running agent for session_id='{session_id}' with query='{query}'")
# #     config = {"configurable": {"thread_id": session_id}}  # For session threading
# #     state = AgentState(query=query, session_id=session_id)
# #     result = app.invoke(state, config=config)
# #     logger.info(f"Agent result %%%% %%% *****: {result.output}")
# #     # Expiry/limit handled by checkpointer; no manual save needed
# #     return result.output


# def run_agent(query: str, session_id: str) -> Dict[str, Any]:
#     logger.info(f"Running agent for session_id='{session_id}' with query='{query}'")
#     config = {"configurable": {"thread_id": session_id}}  # For session threading
#     state = AgentState(query=query, session_id=session_id)

#     try:
#         result = app.invoke(state, config=config)
#         logger.info(f"Agent result %%%% %%% *****: {result.output}")
#         return result.output

#     except Exception as e:
#         logger.exception(f"Agent run failed for session_id='{session_id}' with query='{query}'. Error: {str(e)}")

#         return {
#             "use_case": [],
#             "case_study": [],
#             "insights": [],
#             "message": "Something went wrong while processing your query. Please try again later or contact support."
#         }



# agent.py
"""
LangGraph agent for orchestrating query processing.
Nodes: Input, Routing, Retrieval, Generation, Output.
Session memory via SqliteSaver with expiry/limit.
"""
import os
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver
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
MEMORY_PATH = ":memory:"  # Use in-memory for dev; switch to file for persistence
MEMORY_EXPIRY = 600  # 10 minutes
MEMORY_LIMIT = 3  # Last 3 interactions

# saver = SqliteSaver.from_conn_string(MEMORY_PATH)
saver = MemorySaver()

class AgentState(BaseModel):
    query: str
    session_id: str
    memory: List[Dict] = Field(default_factory=list)  # Previous queries/results
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
ROUTING_PROMPT = """Analyze query: {query} with past context: {memory}. Route to buckets: services, case-studies, insights. Map to funnel stages if similarity > 0.4. Output structured JSON."""
GENERATION_PROMPT = """Given context: {context}, funnel_stage: {funnel_stage}, query: {query}, and past context: {memory}, generate structured JSON with:
- use_case: array of objects with title, url, category
- case_study: array of objects with title, url, category  
- insights: array of objects with title, url, category
- message: string summary
- Avoid using curly or typographic quotes; use straight quotes (') only.

Extract title, url, and category from the context metadata. If no confident match (similarity < 0.4), return empty arrays and acknowledge in message with suggestions for related services."""

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
    memory_str = str(state.memory[-MEMORY_LIMIT:]) if state.memory else ""
    logger.info(f"Routing node: query='{state.query}', memory='{memory_str}'")
    resp = routing_llm.invoke(ROUTING_PROMPT.format(query=state.query, memory=memory_str))
    logger.info(f"Routing LLM output####: {resp.dict()}")
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
        logger.info(f"Retrieved for bucket '{bucket}': {res['matches']}")
        
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
    memory_str = str(state.memory[-MEMORY_LIMIT:]) if state.memory else ""
    
    logger.info(f"Generation node: context={context_str}, funnel_stage={state.funnel_stage}, query={state.query}")
    
    resp = generation_llm.invoke(GENERATION_PROMPT.format(
        context=context_str, 
        funnel_stage=state.funnel_stage, 
        query=state.query, 
        memory=memory_str
    ))
    logger.info(f"Generation LLM output: {resp.dict()}")
    state.output = resp.dict()

    # If no confident matches, keep LLM's message but ensure arrays are empty
    if not any(context.values()):  # No confident matches
        logger.warning("No confident matches found in context.")
        state.output["use_case"] = []
        state.output["case_study"] = []
        state.output["insights"] = []
    # The message from the LLM is preserved, even for no-match cases
    
    return state

def output_node(state: AgentState) -> AgentState:
    logger.info(f"Output node: output={state.output}")
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
    state = AgentState(query=query, session_id=session_id)

    try:
        result = app.invoke(state, config=config)
        logger.info(f"Agent result %%%% %%% *****: {result['output']}")
        return result['output']

    except Exception as e:
        logger.exception(f"Agent run failed for session_id='{session_id}' with query='{query}'. Error: {str(e)}")

        return {
            "use_case": [],
            "case_study": [],
            "insights": [],
            "message": "Something went wrong while processing your query. Please try again later or contact support."
        }