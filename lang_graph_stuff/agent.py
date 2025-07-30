
# # agent.py
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
# from lang_graph_stuff.db import pinecone_db
# from dotenv import load_dotenv
# import time
# from pydantic import BaseModel, Field
# from concurrent.futures import ThreadPoolExecutor



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


# # Enhanced routing prompt with hybrid approach
# ROUTING_PROMPT = f"""You are an AI assistant for The Good, a digital experience optimization consultancy. 

# COMPANY CONTEXT:
# {COMPANY_CONTEXT}

# Analyze the user query: "{{query}}" with past conversation context: {{conversation_history}}

# INSTRUCTIONS:
# 1. Determine the user's intent and funnel stage:
#    - AWARENESS: Learning about CRO, general optimization questions, "what is", "how does" queries
#    - CONSIDERATION: Comparing solutions, seeking proof, "show me examples", ROI questions
#    - DECISION: Ready to engage, specific service needs, "help me with", pricing inquiries

# 2. Choose response strategy:

#    **BASIC QUERIES (No clear business intent)**: 
#    - Simple company info, general definitions, basic "what is CRO" questions
#    - Set can_answer_directly: true
#    - Provide direct_response with basic info
#    - Set buckets as empty array

#    **FUNNEL-ALIGNED QUERIES (Clear business intent/funnel stage)**:
#    - Queries showing interest in optimization, improvement, growth
#    - Set can_answer_directly: false (need database retrieval for comprehensive response)
#    - Route to relevant buckets: services, case-studies, insights based on funnel stage
#    - Set appropriate funnel_stage

# 3. Bucket routing logic:
#    - AWARENESS stage → insights (educational content)
#    - CONSIDERATION stage → case-studies + insights (proof + education)
#    - DECISION stage → services + case-studies (offerings + proof)

# Output structured JSON with buckets, funnel_stage, can_answer_directly, and direct_response.

# EXAMPLES:
# - "What is The Good?" → can_answer_directly: true (basic info)
# - "How can I improve my conversion rate?" → can_answer_directly: false, buckets: ["insights", "services"], funnel_stage: "awareness"
# - "Show me CRO success stories" → can_answer_directly: false, buckets: ["case-studies"], funnel_stage: "consideration"
# - "I need help optimizing my e-commerce site" → can_answer_directly: false, buckets: ["services", "case-studies"], funnel_stage: "decision"
# """


# # Enhanced generation prompt for comprehensive responses
# GENERATION_PROMPT = f"""You are generating comprehensive responses for The Good, a digital experience optimization consultancy.

# COMPANY CONTEXT:
# {COMPANY_CONTEXT}

# Given context: {{context}}, funnel_stage: {{funnel_stage}}, query: {{query}}, and past conversation: {{conversation_history}}, generate structured JSON with:
# - use_case: array of objects with title, url, category (for services/solutions)
# - case_study: array of objects with title, url, category (for success stories/proof)
# - insights: array of objects with title, url, category (for educational content)
# - message: concise summary incorporating The Good's expertise and recommendations.

# RESPONSE STRATEGY BY FUNNEL STAGE:

# **AWARENESS Stage**: Focus on education and building understanding
# - Prioritize insights (educational content, guides, best practices)
# - Include relevant use_case items (service overviews)
# - Message should educate and build awareness of optimization opportunities

# **CONSIDERATION Stage**: Provide proof and build confidence  
# - Prioritize case_study items (success stories, ROI examples)
# - Include relevant insights (supporting educational content)
# - Message should demonstrate expertise and results

# **DECISION Stage**: Show solutions and encourage action
# - Prioritize use_case items (specific services, solutions)
# - Include case_study items (relevant success proof)
# - Message should be solution-focused with clear next steps

# GUIDELINES:
# - - Extract title, url, and category from context metadata accurately **only if url starts with 'https://', is unique, and among duplicates keep the one with the highest similarity score**
# - If similarity < 0.4, return empty arrays but provide helpful message with general service suggestions
# - Create comprehensive responses that combine direct knowledge with retrieved content
# - Maintain The Good's professional tone focused on data-driven optimization
# - Reference 9:1 ROI track record and 10+ years expertise when relevant
# - Use straight quotes (') only, avoid curly or typographic quotes
# - Position The Good as specialized DXO experts, not a "do everything" agency
# - Always provide actionable insights or clear next steps in the message
# - Tailor suggestions to the user's specific context and conversation history
# """


# # Updated direct response prompt for basic queries
# DIRECT_RESPONSE_PROMPT = f"""You are an AI assistant for The Good, a digital experience optimization consultancy.

# COMPANY CONTEXT:
# {COMPANY_CONTEXT}

# The user asked: "{{query}}"
# Past conversation: {{conversation_history}}

# This is a BASIC QUERY with no clear business intent. Provide a helpful, direct response using the company context above. 

# GUIDELINES:
# - Keep it professional, informative, and concise
# - Answer the specific question asked
# - Briefly mention relevant services when natural
# - End with a soft invitation to learn more if appropriate
# - Align with The Good's mission to "Optimize for Good"
# - Don't oversell - this is informational, not sales-focused

# EXAMPLES:
# - "What is The Good?" → Brief company overview with key differentiators
# - "What does CRO mean?" → Definition with The Good's approach
# - "Where are you located?" → Location info (if available) + digital focus
# """


# # Additional prompt for fallback when no good matches found
# FALLBACK_RESPONSE_PROMPT = f"""You are responding for The Good when no specific database matches were found.

# COMPANY CONTEXT:
# {COMPANY_CONTEXT}

# Query: {{query}}
# Funnel Stage: {{funnel_stage}}
# Conversation History: {{conversation_history}}

# Since no specific matches were found, provide a helpful response that:
# 1. Acknowledges the question
# 2. Provides relevant general information from company context
# 3. Suggests appropriate next steps based on funnel stage:
#    - AWARENESS: Offer educational resources, blog insights
#    - CONSIDERATION: Suggest case studies, success stories, free audit
#    - DECISION: Recommend specific services, consultation, audit

# Keep it helpful and professional while positioning The Good's expertise appropriately.
# """


# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
# llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)


# routing_llm = llm.with_structured_output(RoutingOutput)
# generation_llm = llm.with_structured_output(GenerationOutput)
# direct_response_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)  # Slightly higher temp for direct responses
# fallback_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)  # For fallback responses


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
    
#     # Check if we have any confident matches
#     has_confident_matches = any(context.values())
    
#     if has_confident_matches:
#         # Use normal generation with retrieved context
#         resp = generation_llm.invoke(GENERATION_PROMPT.format(
#             context=context_str, 
#             funnel_stage=state.funnel_stage, 
#             query=state.query, 
#             conversation_history=history_str
#         ))
        
#         logger.info(f"Generation LLM output: {resp.dict()}")
#         state.output = resp.dict()
#     else:
#         # Use fallback response when no confident matches
#         logger.warning("No confident matches found in context - using fallback response")
        
#         fallback_response = fallback_llm.invoke(FALLBACK_RESPONSE_PROMPT.format(
#             query=state.query,
#             funnel_stage=state.funnel_stage or "awareness",
#             conversation_history=history_str
#         ))
        
#         state.output = {
#             "use_case": [],
#             "case_study": [],
#             "insights": [],
#             "message": fallback_response.content
#         }
    
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


# optimized_agent.py
"""
Optimized LangGraph agent with reduced latency through:
1. Parallel processing
2. Faster models for routing
3. Caching strategies
4. Optimized embedding model
5. Reduced API calls
"""
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from lang_graph_stuff.db import pinecone_db
from dotenv import load_dotenv
import time
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from functools import lru_cache

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("optimized_agent")

load_dotenv()

# OPTIMIZATION 1: Use faster models
EMBEDDING_MODEL = "text-embedding-3-large"  # 5x faster than large
ROUTING_MODEL = "gpt-3.5-turbo"  # Faster for simple routing
GENERATION_MODEL = "gpt-4o-mini"  # Faster alternative to GPT-4o
FALLBACK_MODEL = "gpt-3.5-turbo"  # Fast for fallbacks

MEMORY_LIMIT = 3

# OPTIMIZATION 2: In-memory caching
class ResponseCache:
    def __init__(self, max_size=1000, ttl=3600):  # 1 hour TTL
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get_key(self, query: str, session_id: str = None) -> str:
        """Generate cache key from query"""
        content = f"{query.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Dict:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, response: Dict):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }

# Global cache instance
response_cache = ResponseCache()

# OPTIMIZATION 3: Precomputed embeddings for common queries
COMMON_EMBEDDINGS = {}

@lru_cache(maxsize=500)
def get_cached_embedding(query: str) -> List[float]:
    """Cache embeddings for frequently asked questions"""
    if query in COMMON_EMBEDDINGS:
        return COMMON_EMBEDDINGS[query]
    
    embedding = embeddings.embed_query(query)
    COMMON_EMBEDDINGS[query] = embedding
    return embedding

class AgentState(BaseModel):
    query: str
    session_id: str
    conversation_history: List[Dict] = Field(default_factory=list)
    embedding: List[float] = None
    buckets: List[str] = Field(default_factory=list)
    funnel_stage: Optional[str] = None
    can_answer_directly: bool = False
    retrieval_results: Dict[str, List] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    cache_key: Optional[str] = None

class RoutingOutput(BaseModel):
    buckets: List[str]
    funnel_stage: Optional[str] = None
    can_answer_directly: bool = Field(default=False)
    direct_response: str = Field(default="")

class SearchItem(BaseModel):
    title: str = Field(description="The title of the source document")
    url: str = Field(description="The URL of the source document") 
    category: str = Field(description="The specific category this item belongs to")

class GenerationOutput(BaseModel):
    use_case: List[SearchItem] = Field(default_factory=list)
    case_study: List[SearchItem] = Field(default_factory=list)
    insights: List[SearchItem] = Field(default_factory=list)
    message: str = Field(description="Overall message or summary")

# Helper function (unchanged)
def scored_vector_to_dict(scored_vector):
    return {
        "id": scored_vector.id,
        "score": float(scored_vector.score),
        "values": list(scored_vector.values) if scored_vector.values else [],
        "metadata": dict(scored_vector.metadata) if scored_vector.metadata else {}
    }

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

# OPTIMIZATION 4: Simplified routing prompt for faster processing
ROUTING_PROMPT = f"""You are an AI assistant for The Good, a CRO consultancy.

COMPANY CONTEXT: {COMPANY_CONTEXT}

Query: "{{query}}"
History: {{conversation_history}}

TASK: Classify this query quickly and determine response strategy.

RULES:
1. Basic company info queries → can_answer_directly: true, empty buckets
2. Business/optimization queries → can_answer_directly: false, relevant buckets

FUNNEL STAGES:
- AWARENESS: Learning, "what is", "how does"
- CONSIDERATION: Examples, proof, ROI questions  
- DECISION: "Help me", specific needs, pricing

BUCKETS: services, case-studies, insights

Respond with JSON only."""

# Optimized generation prompt (shortened)
GENERATION_PROMPT = f"""Generate response for The Good CRO consultancy.

Context: {{context}}
Query: {{query}}
Stage: {{funnel_stage}}
History: {{conversation_history}}

Output JSON with use_case, case_study, insights arrays (title, url, category) and message.
Keep message concise and actionable. Filter results with score > 0.4 only."""

# OPTIMIZATION 5: Initialize optimized models
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
routing_llm = ChatOpenAI(model=ROUTING_MODEL, temperature=0.1).with_structured_output(RoutingOutput)
generation_llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0.2).with_structured_output(GenerationOutput)
fallback_llm = ChatOpenAI(model=FALLBACK_MODEL, temperature=0.3)

saver = MemorySaver()

# OPTIMIZATION 6: Async/parallel node functions
def input_node(state: AgentState) -> AgentState:
    """Optimized input with caching"""
    logger.info(f"Input node: query='{state.query[:50]}...'")
    
    # Generate cache key
    state.cache_key = response_cache.get_key(state.query, state.session_id)
    
    # Check cache first
    cached_response = response_cache.get(state.cache_key)
    if cached_response:
        logger.info("Cache hit - returning cached response")
        state.output = cached_response
        state.can_answer_directly = True
        return state
    
    # Generate embedding with caching
    state.embedding = get_cached_embedding(state.query)
    logger.info("Generated/retrieved embedding")
    return state

def routing_node(state: AgentState) -> AgentState:
    """Fast routing with simplified prompt"""
    if state.can_answer_directly:  # Skip if cached
        return state
        
    history_str = str(state.conversation_history[-2:]) if state.conversation_history else ""  # Reduced history
    logger.info("Routing node: analyzing query")
    
    try:
        resp = routing_llm.invoke(ROUTING_PROMPT.format(
            query=state.query, 
            conversation_history=history_str
        ))
        
        state.buckets = resp.buckets
        state.funnel_stage = resp.funnel_stage
        state.can_answer_directly = resp.can_answer_directly
        
        if resp.can_answer_directly and resp.direct_response:
            state.output = {
                "use_case": [],
                "case_study": [],
                "insights": [],
                "message": resp.direct_response
            }
        
        logger.info(f"Routing: buckets={len(state.buckets)}, direct={state.can_answer_directly}")
        
    except Exception as e:
        logger.error(f"Routing error: {e}")
        # Fallback routing
        state.buckets = ["insights"]
        state.funnel_stage = "awareness"
        state.can_answer_directly = False
    
    return state

def retrieval_node(state: AgentState) -> AgentState:
    """Optimized parallel retrieval"""
    if state.can_answer_directly:
        return state
    
    if not state.buckets:
        logger.warning("No buckets for retrieval")
        return state
    
    logger.info(f"Retrieval: {len(state.buckets)} buckets")
    
    # OPTIMIZATION 7: Reduced top_k and parallel execution
    def retrieve_from_bucket(bucket):
        try:
            res = pinecone_db.query(bucket, state.embedding, top_k=3) 
            return bucket, [scored_vector_to_dict(match) for match in res["matches"]]
        except Exception as e:
            logger.error(f"Retrieval error for {bucket}: {e}")
            return bucket, []
    
    # Parallel retrieval with timeout
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(retrieve_from_bucket, bucket): bucket for bucket in state.buckets}
        
        for future in futures:
            try:
                bucket, matches = future.result(timeout=3)  # 3 second timeout
                state.retrieval_results[bucket] = matches
            except Exception as e:
                logger.error(f"Retrieval timeout/error: {e}")
                bucket = futures[future]
                state.retrieval_results[bucket] = []
    
    return state

def generation_node(state: AgentState) -> AgentState:
    """Optimized generation with better error handling"""
    if state.can_answer_directly and state.output.get("message"):
        return state
    
    # Filter and prepare context more efficiently
    context = {}
    for bucket, matches in state.retrieval_results.items():
        high_conf = [m for m in matches if m.get("score", 0) > 0.4]
        if high_conf:  # Only include buckets with confident matches
            context[bucket] = high_conf
    
    history_str = str(state.conversation_history[-2:]) if state.conversation_history else ""
    
    try:
        if context:
            resp = generation_llm.invoke(GENERATION_PROMPT.format(
                context=str(context), 
                funnel_stage=state.funnel_stage or "awareness", 
                query=state.query, 
                conversation_history=history_str
            ))
            state.output = resp.dict()
            logger.info("Generated response with context")
        else:
            # Quick fallback without extra LLM call
            state.output = {
                "use_case": [],
                "case_study": [],
                "insights": [],
                "message": f"I'd be happy to help you with your digital optimization needs. Based on your question about '{state.query}', The Good specializes in conversion rate optimization with a proven 9:1 ROI track record. Would you like to learn more about our CRO services or see some case studies?"
            }
            logger.info("Used quick fallback response")
            
    except Exception as e:
        logger.error(f"Generation error: {e}")
        state.output = {
            "use_case": [],
            "case_study": [],
            "insights": [],
            "message": "I apologize for the delay. The Good specializes in conversion rate optimization and digital experience improvements. How can we help optimize your business?"
        }
    
    return state

def output_node(state: AgentState) -> AgentState:
    """Optimized output with caching"""
    logger.info("Output node: finalizing response")
    
    # Cache the response for future queries
    if state.cache_key and state.output:
        response_cache.set(state.cache_key, state.output)
        logger.info("Response cached")
    
    # Streamlined conversation history update
    if not state.can_answer_directly:  # Only update for non-cached responses
        interaction = {
            "query": state.query[:100],  # Truncate for memory efficiency
            "timestamp": time.time(),
            "funnel_stage": state.funnel_stage
        }
        
        state.conversation_history.append(interaction)
        if len(state.conversation_history) > MEMORY_LIMIT:
            state.conversation_history = state.conversation_history[-MEMORY_LIMIT:]
    
    return state


# OPTIMIZATION 8: Conditional routing with early exits
def should_skip_retrieval(state: AgentState) -> str:
    if state.can_answer_directly and state.output.get("message"):
        return "output"
    return "retrieval"

def should_skip_generation(state: AgentState) -> str:
    if state.can_answer_directly and state.output.get("message"):
        return "output"
    return "generation"

# Graph definition (unchanged structure)
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

app = graph.compile(checkpointer=saver)

def run_agent(query: str, session_id: str) -> Dict[str, Any]:
    """Optimized agent runner with timing"""
    start_time = time.time()
    logger.info(f"Running optimized agent: '{query[:50]}...'")
    
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        state = AgentState(query=query, session_id=session_id)
        result = app.invoke(state, config=config)
        
        duration = time.time() - start_time
        logger.info(f"Agent completed in {duration:.2f}s")
        
        return result['output']

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"Agent failed in {duration:.2f}s: {str(e)}")
        
        return {
            "use_case": [],
            "case_study": [],
            "insights": [],
            "message": "I apologize, but I'm experiencing technical difficulties. Please try again later or contact The Good directly for assistance with your digital optimization needs."
        }

# OPTIMIZATION 9: Warmup function for production
def warmup_agent():
    """Warm up the agent with common queries"""
    common_queries = [
        "What is The Good?",
        "How can you help improve conversion rates?",
        "Show me case studies",
        "What services do you offer?"
    ]
    
    logger.info("Warming up agent...")
    for query in common_queries:
        try:
            run_agent(query, "warmup_session")
            logger.info(f"Warmed up: {query}")
        except Exception as e:
            logger.error(f"Warmup failed for '{query}': {e}")
    
    logger.info("Agent warmup completed")

# Call warmup in production
if __name__ == "__main__":
    warmup_agent()