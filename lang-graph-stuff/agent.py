
# # agent.py
# """
# LangGraph agent for orchestrating query processing.
# Nodes: Input, Routing, Retrieval, Generation, Output.
# Session memory via MemorySaver with conversation history.
# """
# import os
# import logging
# from typing import Dict, Any, List
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from db import pinecone_db
# from dotenv import load_dotenv
# import time
# from pydantic import BaseModel, Field
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# import functools

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
# MEMORY_LIMIT = 3  # Last 3 interactions

# saver = MemorySaver()

# class AgentState(BaseModel):
#     query: str
#     session_id: str
#     conversation_history: List[Dict] = Field(default_factory=list)  # Current conversation context
#     embedding: List[float] = None
#     buckets: List[str] = Field(default_factory=list)
#     funnel_stage: str = None
#     retrieval_results: Dict[str, List] = Field(default_factory=dict)
#     output: Dict[str, Any] = Field(default_factory=dict)

# # Structured output models
# class RoutingOutput(BaseModel):
#     buckets: List[str]
#     funnel_stage: str = None

# class SearchItem(BaseModel):
#     title: str = Field(description="The title of the source document")
#     url: str = Field(description="The URL of the source document") 
#     category: str = Field(description="The specific category or subcategory this item belongs to")

# class GenerationOutput(BaseModel):
#     use_case: List[SearchItem] = Field(default_factory=list, description="List of use case items with title, URL and category")
#     case_study: List[SearchItem] = Field(default_factory=list, description="List of case study items with title, URL and category")
#     insights: List[SearchItem] = Field(default_factory=list, description="List of insight items with title, URL and category")
#     message: str = Field(description="Overall message or summary")

# # Helper function to convert ScoredVector to serializable dict
# def scored_vector_to_dict(scored_vector):
#     """Convert Pinecone ScoredVector to serializable dictionary"""
#     return {
#         "id": scored_vector.id,
#         "score": float(scored_vector.score),
#         "values": list(scored_vector.values) if scored_vector.values else [],
#         "metadata": dict(scored_vector.metadata) if scored_vector.metadata else {}
#     }

# # Prompts (enhanced with memory)
# ROUTING_PROMPT = """Analyze query: {query} with past conversation context: {conversation_history}. 
# Route to relevant buckets from: services, case-studies, insights. 
# Map to funnel stages (awareness, consideration, decision) if similarity > 0.4. 
# Output structured JSON with buckets and funnel_stage."""

# GENERATION_PROMPT = """Given context: {context}, funnel_stage: {funnel_stage}, query: {query}, and past conversation: {conversation_history}, generate structured JSON with:
# - use_case: array of objects with title, url, category
# - case_study: array of objects with title, url, category  
# - insights: array of objects with title, url, category
# - message: string summary
# - Avoid using curly or typographic quotes; use straight quotes (') only.

# Extract title, url, and category from the context metadata. If no confident match (similarity < 0.4), return empty arrays and acknowledge in message with suggestions for related services.
# Consider the conversation history to provide more personalized and contextual responses."""

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
#     # Use conversation history for context
#     history_str = str(state.conversation_history[-MEMORY_LIMIT:]) if state.conversation_history else ""
#     logger.info(f"Routing node: query='{state.query}', conversation_history='{history_str}'")
    
#     resp = routing_llm.invoke(ROUTING_PROMPT.format(
#         query=state.query, 
#         conversation_history=history_str
#     ))
    
#     logger.info(f"Routing LLM output: buckets={resp.buckets}, funnel_stage={resp.funnel_stage}")
#     state.buckets = resp.buckets
#     state.funnel_stage = resp.funnel_stage
#     return state


# def retrieval_node(state: AgentState) -> AgentState:
#     """Parallel retrieval using ThreadPoolExecutor - no async complications"""
#     logger.info(f"Retrieval node: buckets={state.buckets}")
#     if not state.buckets:
#         logger.warning("No buckets found for retrieval.")
#         return state  # Fallback: no retrieval
    
#     # Function to retrieve from a single bucket
#     def retrieve_from_bucket(bucket):
#         try:
#             res = pinecone_db.query(bucket, state.embedding, top_k=3)
#             logger.info(f"Retrieved for bucket '{bucket}': {len(res['matches'])} matches")
            
#             # Convert ScoredVector objects to serializable dictionaries
#             serializable_matches = []
#             for match in res["matches"]:
#                 serializable_matches.append(scored_vector_to_dict(match))
            
#             return bucket, serializable_matches
#         except Exception as e:
#             logger.error(f"Error retrieving from bucket '{bucket}': {str(e)}")
#             return bucket, []
    
#     # Use ThreadPoolExecutor to run queries in parallel
#     with ThreadPoolExecutor(max_workers=min(len(state.buckets), 5)) as executor:
#         # Submit all bucket queries
#         future_to_bucket = {
#             executor.submit(retrieve_from_bucket, bucket): bucket 
#             for bucket in state.buckets
#         }
        
#         # Collect results as they complete
#         for future in future_to_bucket:
#             bucket, matches = future.result()
#             state.retrieval_results[bucket] = matches
    
#     return state



# def generation_node(state: AgentState) -> AgentState:
#     # Check confidence (filter low-similarity results)
#     context = {}
#     for bucket, matches in state.retrieval_results.items():
#         high_conf = [m for m in matches if m["score"] > 0.4]
#         context[bucket] = high_conf or []
    
#     context_str = str(context)
#     history_str = str(state.conversation_history[-MEMORY_LIMIT:]) if state.conversation_history else ""
    
#     logger.info(f"Generation node: query={state.query}, funnel_stage={state.funnel_stage}")
    
#     resp = generation_llm.invoke(GENERATION_PROMPT.format(
#         context=context_str, 
#         funnel_stage=state.funnel_stage, 
#         query=state.query, 
#         conversation_history=history_str
#     ))
    
#     logger.info(f"Generation LLM output: {resp.dict()}")
#     state.output = resp.dict()

#     # If no confident matches, keep LLM's message but ensure arrays are empty
#     if not any(context.values()):  # No confident matches
#         logger.warning("No confident matches found in context.")
#         state.output["use_case"] = []
#         state.output["case_study"] = []
#         state.output["insights"] = []
    
#     return state

# def output_node(state: AgentState) -> AgentState:
#     logger.info(f"Output node: output={state.output}")
    
#     # Update conversation history with this interaction
#     interaction = {
#         "query": state.query,
#         "response": state.output,
#         "timestamp": time.time(),
#         "funnel_stage": state.funnel_stage,
#         "buckets": state.buckets
#     }
    
#     # Add to conversation history and maintain limit
#     state.conversation_history.append(interaction)
#     if len(state.conversation_history) > MEMORY_LIMIT:
#         state.conversation_history = state.conversation_history[-MEMORY_LIMIT:]
    
#     logger.info(f"Updated conversation history: {len(state.conversation_history)} interactions")
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

# def run_agent(query: str, session_id: str) -> Dict[str, Any]:
#     logger.info(f"Running agent for session_id='{session_id}' with query='{query}'")
#     config = {"configurable": {"thread_id": session_id}}  # For session threading
    
#     try:
#         # Get previous state from checkpointer to maintain conversation history
#         state = AgentState(query=query, session_id=session_id)
        
#         result = app.invoke(state, config=config)
#         logger.info(f"Agent completed successfully")
#         return result['output']

#     except Exception as e:
#         logger.exception(f"Agent run failed for session_id='{session_id}' with query='{query}'. Error: {str(e)}")

#         return {
#             "use_case": [],
#             "case_study": [],
#             "insights": [],
#             "message": "Something went wrong while processing your query. Please try again later or contact support."
#         }


# agent.py

# """
# LangGraph agent for orchestrating query processing for The Good.
# Nodes: Input, Routing, Retrieval, Generation, Output.
# Session memory via MemorySaver with conversation history.
# Enhanced with company context and direct response capability.
# """
# import os
# import logging
# from typing import Dict, Any, List
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from db import pinecone_db
# from dotenv import load_dotenv
# import time
# from pydantic import BaseModel, Field
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# import functools


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
# MEMORY_LIMIT = 3  # Last 3 interactions


# saver = MemorySaver()


# class AgentState(BaseModel):
#     query: str
#     session_id: str
#     conversation_history: List[Dict] = Field(default_factory=list)  # Current conversation context
#     embedding: List[float] = None
#     buckets: List[str] = Field(default_factory=list)
#     funnel_stage: str = None
#     can_answer_directly: bool = False  # New field for direct responses
#     retrieval_results: Dict[str, List] = Field(default_factory=dict)
#     output: Dict[str, Any] = Field(default_factory=dict)


# # Structured output models
# class RoutingOutput(BaseModel):
#     buckets: List[str]
#     funnel_stage: str = None
#     can_answer_directly: bool = Field(default=False, description="Whether the query can be answered directly without database retrieval")
#     direct_response: str = Field(default="", description="Direct answer if can_answer_directly is True")


# class SearchItem(BaseModel):
#     title: str = Field(description="The title of the source document")
#     url: str = Field(description="The URL of the source document") 
#     category: str = Field(description="The specific category or subcategory this item belongs to")


# class GenerationOutput(BaseModel):
#     use_case: List[SearchItem] = Field(default_factory=list, description="List of use case items with title, URL and category")
#     case_study: List[SearchItem] = Field(default_factory=list, description="List of case study items with title, URL and category")
#     insights: List[SearchItem] = Field(default_factory=list, description="List of insight items with title, URL and category")
#     message: str = Field(description="Overall message or summary")


# # Helper function to convert ScoredVector to serializable dict
# def scored_vector_to_dict(scored_vector):
#     """Convert Pinecone ScoredVector to serializable dictionary"""
#     return {
#         "id": scored_vector.id,
#         "score": float(scored_vector.score),
#         "values": list(scored_vector.values) if scored_vector.values else [],
#         "metadata": dict(scored_vector.metadata) if scored_vector.metadata else {}
#     }


# # Company Context for The Good
# COMPANY_CONTEXT = """
# THE GOOD - COMPANY OVERVIEW:
# The Good is a specialized digital experience optimization (DXO) consultancy that helps e-commerce brands and SaaS companies improve their online performance and grow revenue. We focus exclusively on conversion rate optimization (CRO), not as a "do everything" agency.

# KEY SERVICES:
# • Conversion Rate Optimization (CRO) with 9:1 ROI track record
# • Digital Experience Optimization Program™ (DXO)
# • A/B Testing & Experimentation Programs
# • User Experience (UX) Research & Design
# • Analytics Setup & Data-Driven Strategy
# • Personalization & Customer Journey Optimization
# • CRO Audits & Optimization Roadmaps

# SPECIALIZATIONS:
# • E-commerce conversion optimization
# • SaaS retention improvement (making products "cancel-proof")
# • ROAS improvement for paid traffic
# • User experience enhancement
# • Customer behavior analysis

# MISSION & VALUES:
# "Remove all the bad digital experiences until only the good remain" - We focus on ethical, sustainable growth through data-driven optimization. Our tagline is "Optimize for Good."

# TYPICAL CLIENTS:
# Mid-to-large enterprises in retail, fashion, health, SaaS, and e-commerce sectors. We've worked with major brands to increase conversion rates by 20-50% through targeted optimizations.

# FUNNEL ALIGNMENT:
# • Awareness: Educational insights, free resources, webinars
# • Consideration: Case studies, client success stories, ROI proof
# • Decision: Detailed service offerings, optimization programs, audits

# 10+ years of CRO research and strategy expertise with proven methodologies.
# """


# # Enhanced prompts with company context
# ROUTING_PROMPT = f"""You are an AI assistant for The Good, a digital experience optimization consultancy. 

# COMPANY CONTEXT:
# {COMPANY_CONTEXT}

# Analyze the user query: "{{query}}" with past conversation context: {{conversation_history}}

# INSTRUCTIONS:
# 1. First determine if you can answer the query directly using the company context above (basic company info, services overview, mission, general CRO questions, etc.)

# 2. If you CAN answer directly:
#    - Set can_answer_directly: true
#    - Provide a helpful direct_response using the company context
#    - Set buckets as empty array
#    - Set funnel_stage if applicable

# 3. If you CANNOT answer directly (need specific case studies, detailed insights, or complex technical information):
#    - Set can_answer_directly: false
#    - Route to relevant buckets from: services, case-studies, insights
#    - Map to funnel stages (awareness, consideration, decision) based on intent
#    - Leave direct_response empty

# Output structured JSON with buckets, funnel_stage, can_answer_directly, and direct_response.
# """


# GENERATION_PROMPT = f"""You are generating responses for The Good, a digital experience optimization consultancy.

# COMPANY CONTEXT:
# {COMPANY_CONTEXT}

# Given context: {{context}}, funnel_stage: {{funnel_stage}}, query: {{query}}, and past conversation: {{conversation_history}}, generate structured JSON with:
# - use_case: array of objects with title, url, category
# - case_study: array of objects with title, url, category  
# - insights: array of objects with title, url, category
# - message: string summary incorporating The Good's expertise and approach

# GUIDELINES:
# - Extract title, url, and category from the context metadata
# - If no confident match (similarity < 0.4), return empty arrays and acknowledge in message with suggestions for The Good's related services
# - Consider conversation history for personalized, contextual responses
# - Maintain The Good's professional tone focused on data-driven optimization and ethical growth
# - Reference our 9:1 ROI track record and 10+ years expertise when relevant
# - Use straight quotes (') only, avoid curly or typographic quotes
# - Position The Good as the specialized DXO experts, not a "do everything" agency
# """


# # Direct response prompt for basic queries
# DIRECT_RESPONSE_PROMPT = f"""You are an AI assistant for The Good, a digital experience optimization consultancy.

# COMPANY CONTEXT:
# {COMPANY_CONTEXT}

# The user asked: "{{query}}"
# Past conversation: {{conversation_history}}

# Provide a helpful, direct response using the company context above. Keep it professional, informative, and aligned with The Good's mission to "Optimize for Good." Mention relevant services or invite them to learn more when appropriate.
# """


# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
# llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)


# routing_llm = llm.with_structured_output(RoutingOutput)
# generation_llm = llm.with_structured_output(GenerationOutput)
# direct_response_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)  # Slightly higher temp for direct responses


# # --- Enhanced Node functions ---
# def input_node(state: AgentState) -> AgentState:
#     logger.info(f"Input node: query='{state.query}', session_id='{state.session_id}'")
#     state.embedding = embeddings.embed_query(state.query)
#     logger.info(f"Generated embedding: {state.embedding[:5]}...")  # Show only first 5 values
#     return state


# def routing_node(state: AgentState) -> AgentState:
#     # Use conversation history for context
#     history_str = str(state.conversation_history[-MEMORY_LIMIT:]) if state.conversation_history else ""
#     logger.info(f"Routing node: query='{state.query}', conversation_history='{history_str}'")
    
#     resp = routing_llm.invoke(ROUTING_PROMPT.format(
#         query=state.query, 
#         conversation_history=history_str
#     ))
#     logger.info(f"Routing LLM output**********: {resp.dict()}")
#     logger.info(f"Routing LLM output: buckets={resp.buckets}, funnel_stage={resp.funnel_stage}, can_answer_directly={resp.can_answer_directly}")
    
#     state.buckets = resp.buckets
#     state.funnel_stage = resp.funnel_stage
#     state.can_answer_directly = resp.can_answer_directly
    
#     # If we can answer directly, set the response
#     if resp.can_answer_directly and resp.direct_response:
#         state.output = {
#             "use_case": [],
#             "case_study": [],
#             "insights": [],
#             "message": resp.direct_response
#         }
    
#     return state


# def retrieval_node(state: AgentState) -> AgentState:
#     """Parallel retrieval using ThreadPoolExecutor - skip if direct answer available"""
#     if state.can_answer_directly:
#         logger.info("Skipping retrieval - direct answer available")
#         return state
    
#     logger.info(f"Retrieval node: buckets={state.buckets}")
#     if not state.buckets:
#         logger.warning("No buckets found for retrieval.")
#         return state  # Fallback: no retrieval
    
#     # Function to retrieve from a single bucket
#     def retrieve_from_bucket(bucket):
#         try:
#             res = pinecone_db.query(bucket, state.embedding, top_k=3)
#             logger.info(f"Retrieved for bucket '{bucket}': {len(res['matches'])} matches")
            
#             # Convert ScoredVector objects to serializable dictionaries
#             serializable_matches = []
#             for match in res["matches"]:
#                 serializable_matches.append(scored_vector_to_dict(match))
            
#             return bucket, serializable_matches
#         except Exception as e:
#             logger.error(f"Error retrieving from bucket '{bucket}': {str(e)}")
#             return bucket, []
    
#     # Use ThreadPoolExecutor to run queries in parallel
#     with ThreadPoolExecutor(max_workers=min(len(state.buckets), 5)) as executor:
#         # Submit all bucket queries
#         future_to_bucket = {
#             executor.submit(retrieve_from_bucket, bucket): bucket 
#             for bucket in state.buckets
#         }
        
#         # Collect results as they complete
#         for future in future_to_bucket:
#             bucket, matches = future.result()
#             state.retrieval_results[bucket] = matches
    
#     return state


# def generation_node(state: AgentState) -> AgentState:
#     """Generate response - skip if direct answer already available"""
#     if state.can_answer_directly and state.output.get("message"):
#         logger.info("Skipping generation - direct answer already available")
#         return state
    
#     # Check confidence (filter low-similarity results)
#     context = {}
#     for bucket, matches in state.retrieval_results.items():
#         high_conf = [m for m in matches if m["score"] > 0.4]
#         context[bucket] = high_conf or []
    
#     context_str = str(context)
#     history_str = str(state.conversation_history[-MEMORY_LIMIT:]) if state.conversation_history else ""
    
#     logger.info(f"Generation node: query={state.query}, funnel_stage={state.funnel_stage}")
    
#     resp = generation_llm.invoke(GENERATION_PROMPT.format(
#         context=context_str, 
#         funnel_stage=state.funnel_stage, 
#         query=state.query, 
#         conversation_history=history_str
#     ))
    
#     logger.info(f"Generation LLM output: {resp.dict()}")
#     state.output = resp.dict()

#     # If no confident matches, keep LLM's message but ensure arrays are empty
#     if not any(context.values()):  # No confident matches
#         logger.warning("No confident matches found in context.")
#         state.output["use_case"] = []
#         state.output["case_study"] = []
#         state.output["insights"] = []
    
#     return state


# def output_node(state: AgentState) -> AgentState:
#     logger.info(f"Output node: output={state.output}")
    
#     # Update conversation history with this interaction
#     interaction = {
#         "query": state.query,
#         "response": state.output,
#         "timestamp": time.time(),
#         "funnel_stage": state.funnel_stage,
#         "buckets": state.buckets,
#         "direct_answer": state.can_answer_directly
#     }
    
#     # Add to conversation history and maintain limit
#     state.conversation_history.append(interaction)
#     if len(state.conversation_history) > MEMORY_LIMIT:
#         state.conversation_history = state.conversation_history[-MEMORY_LIMIT:]
    
#     logger.info(f"Updated conversation history: {len(state.conversation_history)} interactions")
#     return state


# # --- Graph definition with conditional routing ---
# def should_skip_retrieval(state: AgentState) -> str:
#     """Conditional routing: skip retrieval and generation if we can answer directly"""
#     if state.can_answer_directly and state.output.get("message"):
#         return "output"
#     return "retrieval"


# def should_skip_generation(state: AgentState) -> str:
#     """Conditional routing: skip generation if direct answer already available"""
#     if state.can_answer_directly and state.output.get("message"):
#         return "output"
#     return "generation"


# graph = StateGraph(AgentState)
# graph.add_node("input", input_node)
# graph.add_node("routing", routing_node)
# graph.add_node("retrieval", retrieval_node)
# graph.add_node("generation", generation_node)
# graph.add_node("output", output_node)


# graph.add_edge("input", "routing")
# graph.add_conditional_edges("routing", should_skip_retrieval, {"retrieval": "retrieval", "output": "output"})
# graph.add_conditional_edges("retrieval", should_skip_generation, {"generation": "generation", "output": "output"})
# graph.add_edge("generation", "output")
# graph.add_edge("output", END)


# graph.set_entry_point("input")


# app = graph.compile(checkpointer=saver)  # Compile with checkpointer


# def run_agent(query: str, session_id: str) -> Dict[str, Any]:
#     logger.info(f"Running agent for session_id='{session_id}' with query='{query}'")
#     config = {"configurable": {"thread_id": session_id}}  # For session threading
    
#     try:
#         # Get previous state from checkpointer to maintain conversation history
#         state = AgentState(query=query, session_id=session_id)
        
#         result = app.invoke(state, config=config)
#         logger.info(f"Agent completed successfully")
#         return result['output']

#     except Exception as e:
#         logger.exception(f"Agent run failed for session_id='{session_id}' with query='{query}'. Error: {str(e)}")

#         return {
#             "use_case": [],
#             "case_study": [],
#             "insights": [],
#             "message": "I apologize, but I'm experiencing technical difficulties. Please try again later or contact The Good directly for assistance with your digital optimization needs."
#         }







# agent.py
"""
LangGraph agent for orchestrating query processing for The Good.
Nodes: Input, Routing, Retrieval, Generation, Output.
Session memory via MemorySaver with conversation history.
Enhanced with company context and direct response capability.
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools


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
    can_answer_directly: bool = False  # New field for direct responses
    retrieval_results: Dict[str, List] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)


# Structured output models
class RoutingOutput(BaseModel):
    buckets: List[str]
    funnel_stage: str = None
    can_answer_directly: bool = Field(default=False, description="Whether the query can be answered directly without database retrieval")
    direct_response: str = Field(default="", description="Direct answer if can_answer_directly is True")


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


# Company Context for The Good
COMPANY_CONTEXT = """
THE GOOD - COMPANY OVERVIEW:
The Good is a specialized digital experience optimization (DXO) consultancy that helps e-commerce brands and SaaS companies improve their online performance and grow revenue. We focus exclusively on conversion rate optimization (CRO), not as a "do everything" agency.

KEY SERVICES:
• Conversion Rate Optimization (CRO) with 9:1 ROI track record
• Digital Experience Optimization Program™ (DXO)
• A/B Testing & Experimentation Programs
• User Experience (UX) Research & Design
• Analytics Setup & Data-Driven Strategy
• Personalization & Customer Journey Optimization
• CRO Audits & Optimization Roadmaps

SPECIALIZATIONS:
• E-commerce conversion optimization
• SaaS retention improvement (making products "cancel-proof")
• ROAS improvement for paid traffic
• User experience enhancement
• Customer behavior analysis

MISSION & VALUES:
"Remove all the bad digital experiences until only the good remain" - We focus on ethical, sustainable growth through data-driven optimization. Our tagline is "Optimize for Good."

TYPICAL CLIENTS:
Mid-to-large enterprises in retail, fashion, health, SaaS, and e-commerce sectors. We've worked with major brands to increase conversion rates by 20-50% through targeted optimizations.

FUNNEL ALIGNMENT:
• Awareness: Educational insights, free resources, webinars
• Consideration: Case studies, client success stories, ROI proof
• Decision: Detailed service offerings, optimization programs, audits

10+ years of CRO research and strategy expertise with proven methodologies.
"""


# Enhanced routing prompt with hybrid approach
ROUTING_PROMPT = f"""You are an AI assistant for The Good, a digital experience optimization consultancy. 

COMPANY CONTEXT:
{COMPANY_CONTEXT}

Analyze the user query: "{{query}}" with past conversation context: {{conversation_history}}

INSTRUCTIONS:
1. Determine the user's intent and funnel stage:
   - AWARENESS: Learning about CRO, general optimization questions, "what is", "how does" queries
   - CONSIDERATION: Comparing solutions, seeking proof, "show me examples", ROI questions
   - DECISION: Ready to engage, specific service needs, "help me with", pricing inquiries

2. Choose response strategy:

   **BASIC QUERIES (No clear business intent)**: 
   - Simple company info, general definitions, basic "what is CRO" questions
   - Set can_answer_directly: true
   - Provide direct_response with basic info
   - Set buckets as empty array

   **FUNNEL-ALIGNED QUERIES (Clear business intent/funnel stage)**:
   - Queries showing interest in optimization, improvement, growth
   - Set can_answer_directly: false (need database retrieval for comprehensive response)
   - Route to relevant buckets: services, case-studies, insights based on funnel stage
   - Set appropriate funnel_stage

3. Bucket routing logic:
   - AWARENESS stage → insights (educational content)
   - CONSIDERATION stage → case-studies + insights (proof + education)
   - DECISION stage → services + case-studies (offerings + proof)

Output structured JSON with buckets, funnel_stage, can_answer_directly, and direct_response.

EXAMPLES:
- "What is The Good?" → can_answer_directly: true (basic info)
- "How can I improve my conversion rate?" → can_answer_directly: false, buckets: ["insights", "services"], funnel_stage: "awareness"
- "Show me CRO success stories" → can_answer_directly: false, buckets: ["case-studies"], funnel_stage: "consideration"
- "I need help optimizing my e-commerce site" → can_answer_directly: false, buckets: ["services", "case-studies"], funnel_stage: "decision"
"""


# Enhanced generation prompt for comprehensive responses
GENERATION_PROMPT = f"""You are generating comprehensive responses for The Good, a digital experience optimization consultancy.

COMPANY CONTEXT:
{COMPANY_CONTEXT}

Given context: {{context}}, funnel_stage: {{funnel_stage}}, query: {{query}}, and past conversation: {{conversation_history}}, generate structured JSON with:
- use_case: array of objects with title, url, category (for services/solutions)
- case_study: array of objects with title, url, category (for success stories/proof)
- insights: array of objects with title, url, category (for educational content)
- message: comprehensive summary incorporating The Good's expertise

RESPONSE STRATEGY BY FUNNEL STAGE:

**AWARENESS Stage**: Focus on education and building understanding
- Prioritize insights (educational content, guides, best practices)
- Include relevant use_case items (service overviews)
- Message should educate and build awareness of optimization opportunities

**CONSIDERATION Stage**: Provide proof and build confidence  
- Prioritize case_study items (success stories, ROI examples)
- Include relevant insights (supporting educational content)
- Message should demonstrate expertise and results

**DECISION Stage**: Show solutions and encourage action
- Prioritize use_case items (specific services, solutions)
- Include case_study items (relevant success proof)
- Message should be solution-focused with clear next steps

GUIDELINES:
- Extract title, url, and category from context metadata accurately
- If similarity < 0.4, return empty arrays but provide helpful message with general service suggestions
- Create comprehensive responses that combine direct knowledge with retrieved content
- Maintain The Good's professional tone focused on data-driven optimization
- Reference 9:1 ROI track record and 10+ years expertise when relevant
- Use straight quotes (') only, avoid curly or typographic quotes
- Position The Good as specialized DXO experts, not a "do everything" agency
- Always provide actionable insights or clear next steps in the message
- Tailor suggestions to the user's specific context and conversation history
"""


# Updated direct response prompt for basic queries
DIRECT_RESPONSE_PROMPT = f"""You are an AI assistant for The Good, a digital experience optimization consultancy.

COMPANY CONTEXT:
{COMPANY_CONTEXT}

The user asked: "{{query}}"
Past conversation: {{conversation_history}}

This is a BASIC QUERY with no clear business intent. Provide a helpful, direct response using the company context above. 

GUIDELINES:
- Keep it professional, informative, and concise
- Answer the specific question asked
- Briefly mention relevant services when natural
- End with a soft invitation to learn more if appropriate
- Align with The Good's mission to "Optimize for Good"
- Don't oversell - this is informational, not sales-focused

EXAMPLES:
- "What is The Good?" → Brief company overview with key differentiators
- "What does CRO mean?" → Definition with The Good's approach
- "Where are you located?" → Location info (if available) + digital focus
"""


# Additional prompt for fallback when no good matches found
FALLBACK_RESPONSE_PROMPT = f"""You are responding for The Good when no specific database matches were found.

COMPANY CONTEXT:
{COMPANY_CONTEXT}

Query: {{query}}
Funnel Stage: {{funnel_stage}}
Conversation History: {{conversation_history}}

Since no specific matches were found, provide a helpful response that:
1. Acknowledges the question
2. Provides relevant general information from company context
3. Suggests appropriate next steps based on funnel stage:
   - AWARENESS: Offer educational resources, blog insights
   - CONSIDERATION: Suggest case studies, success stories, free audit
   - DECISION: Recommend specific services, consultation, audit

Keep it helpful and professional while positioning The Good's expertise appropriately.
"""


embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)


routing_llm = llm.with_structured_output(RoutingOutput)
generation_llm = llm.with_structured_output(GenerationOutput)
direct_response_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)  # Slightly higher temp for direct responses
fallback_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)  # For fallback responses


# --- Enhanced Node functions ---
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
    logger.info(f"Routing LLM output**********: {resp.dict()}")
    logger.info(f"Routing LLM output: buckets={resp.buckets}, funnel_stage={resp.funnel_stage}, can_answer_directly={resp.can_answer_directly}")
    
    state.buckets = resp.buckets
    state.funnel_stage = resp.funnel_stage
    state.can_answer_directly = resp.can_answer_directly
    
    # If we can answer directly, set the response
    if resp.can_answer_directly and resp.direct_response:
        state.output = {
            "use_case": [],
            "case_study": [],
            "insights": [],
            "message": resp.direct_response
        }
    
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """Parallel retrieval using ThreadPoolExecutor - skip if direct answer available"""
    if state.can_answer_directly:
        logger.info("Skipping retrieval - direct answer available")
        return state
    
    logger.info(f"Retrieval node: buckets={state.buckets}")
    if not state.buckets:
        logger.warning("No buckets found for retrieval.")
        return state  # Fallback: no retrieval
    
    # Function to retrieve from a single bucket
    def retrieve_from_bucket(bucket):
        try:
            res = pinecone_db.query(bucket, state.embedding, top_k=3)
            logger.info(f"Retrieved for bucket '{bucket}': {len(res['matches'])} matches")
            
            # Convert ScoredVector objects to serializable dictionaries
            serializable_matches = []
            for match in res["matches"]:
                serializable_matches.append(scored_vector_to_dict(match))
            
            return bucket, serializable_matches
        except Exception as e:
            logger.error(f"Error retrieving from bucket '{bucket}': {str(e)}")
            return bucket, []
    
    # Use ThreadPoolExecutor to run queries in parallel
    with ThreadPoolExecutor(max_workers=min(len(state.buckets), 5)) as executor:
        # Submit all bucket queries
        future_to_bucket = {
            executor.submit(retrieve_from_bucket, bucket): bucket 
            for bucket in state.buckets
        }
        
        # Collect results as they complete
        for future in future_to_bucket:
            bucket, matches = future.result()
            state.retrieval_results[bucket] = matches
    
    return state


def generation_node(state: AgentState) -> AgentState:
    """Generate response - skip if direct answer already available"""
    if state.can_answer_directly and state.output.get("message"):
        logger.info("Skipping generation - direct answer already available")
        return state
    
    # Check confidence (filter low-similarity results)
    context = {}
    for bucket, matches in state.retrieval_results.items():
        high_conf = [m for m in matches if m["score"] > 0.4]
        context[bucket] = high_conf or []
    
    context_str = str(context)
    history_str = str(state.conversation_history[-MEMORY_LIMIT:]) if state.conversation_history else ""
    
    logger.info(f"Generation node: query={state.query}, funnel_stage={state.funnel_stage}")
    
    # Check if we have any confident matches
    has_confident_matches = any(context.values())
    
    if has_confident_matches:
        # Use normal generation with retrieved context
        resp = generation_llm.invoke(GENERATION_PROMPT.format(
            context=context_str, 
            funnel_stage=state.funnel_stage, 
            query=state.query, 
            conversation_history=history_str
        ))
        
        logger.info(f"Generation LLM output: {resp.dict()}")
        state.output = resp.dict()
    else:
        # Use fallback response when no confident matches
        logger.warning("No confident matches found in context - using fallback response")
        
        fallback_response = fallback_llm.invoke(FALLBACK_RESPONSE_PROMPT.format(
            query=state.query,
            funnel_stage=state.funnel_stage or "awareness",
            conversation_history=history_str
        ))
        
        state.output = {
            "use_case": [],
            "case_study": [],
            "insights": [],
            "message": fallback_response.content
        }
    
    return state


def output_node(state: AgentState) -> AgentState:
    logger.info(f"Output node: output={state.output}")
    
    # Update conversation history with this interaction
    interaction = {
        "query": state.query,
        "response": state.output,
        "timestamp": time.time(),
        "funnel_stage": state.funnel_stage,
        "buckets": state.buckets,
        "direct_answer": state.can_answer_directly
    }
    
    # Add to conversation history and maintain limit
    state.conversation_history.append(interaction)
    if len(state.conversation_history) > MEMORY_LIMIT:
        state.conversation_history = state.conversation_history[-MEMORY_LIMIT:]
    
    logger.info(f"Updated conversation history: {len(state.conversation_history)} interactions")
    return state


# --- Graph definition with conditional routing ---
def should_skip_retrieval(state: AgentState) -> str:
    """Conditional routing: skip retrieval and generation if we can answer directly"""
    if state.can_answer_directly and state.output.get("message"):
        return "output"
    return "retrieval"


def should_skip_generation(state: AgentState) -> str:
    """Conditional routing: skip generation if direct answer already available"""
    if state.can_answer_directly and state.output.get("message"):
        return "output"
    return "generation"


graph = StateGraph(AgentState)
graph.add_node("input", input_node)
graph.add_node("routing", routing_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("generation", generation_node)
graph.add_node("output", output_node)


graph.add_edge("input", "routing")
graph.add_conditional_edges("routing", should_skip_retrieval, {"retrieval": "retrieval", "output": "output"})
graph.add_conditional_edges("retrieval", should_skip_generation, {"generation": "generation", "output": "output"})
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
            "message": "I apologize, but I'm experiencing technical difficulties. Please try again later or contact The Good directly for assistance with your digital optimization needs."
        }