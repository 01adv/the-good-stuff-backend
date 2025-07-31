# """
# Optimized LangGraph agent with reduced latency and SaaS-focused routing through:
# 1. Parallel processing
# 2. Faster models for routing
# 3. Caching strategies
# 4. Optimized embedding model
# 5. Reduced API calls
# 6. SaaS service-centric approach
# """
# import os
# import logging
# import asyncio
# from typing import Dict, Any, List, Optional
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from lang_graph_stuff.db import pinecone_db
# from dotenv import load_dotenv
# import time
# from pydantic import BaseModel, Field
# from concurrent.futures import ThreadPoolExecutor
# import hashlib
# import json
# from functools import lru_cache

# # --- Logging setup ---
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# logger = logging.getLogger("optimized_agent")

# load_dotenv()

# # OPTIMIZATION 1: Use faster models
# EMBEDDING_MODEL = "text-embedding-3-large" 
# ROUTING_MODEL = "gpt-3.5-turbo"  # Faster for simple routing
# GENERATION_MODEL = "gpt-4o-mini"  # Faster alternative to GPT-4o
# FALLBACK_MODEL = "gpt-3.5-turbo"  # Fast for fallbacks

# MEMORY_LIMIT = 3

# # OPTIMIZATION 2: In-memory caching
# class ResponseCache:
#     def __init__(self, max_size=1000, ttl=3600):  # 1 hour TTL
#         self.cache = {}
#         self.max_size = max_size
#         self.ttl = ttl
    
#     def get_key(self, query: str, session_id: str = None) -> str:
#         """Generate cache key from query"""
#         content = f"{query.lower().strip()}"
#         return hashlib.md5(content.encode()).hexdigest()[:16]
    
#     def get(self, key: str) -> Dict:
#         if key in self.cache:
#             entry = self.cache[key]
#             if time.time() - entry['timestamp'] < self.ttl:
#                 return entry['response']
#             else:
#                 del self.cache[key]
#         return None
    
#     def set(self, key: str, response: Dict):
#         if len(self.cache) >= self.max_size:
#             # Remove oldest entry
#             oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
#             del self.cache[oldest_key]
        
#         self.cache[key] = {
#             'response': response,
#             'timestamp': time.time()
#         }

# # Global cache instance
# response_cache = ResponseCache()

# # OPTIMIZATION 3: Precomputed embeddings for common queries
# COMMON_EMBEDDINGS = {}

# @lru_cache(maxsize=500)
# def get_cached_embedding(query: str) -> List[float]:
#     """Cache embeddings for frequently asked questions"""
#     if query in COMMON_EMBEDDINGS:
#         return COMMON_EMBEDDINGS[query]
    
#     embedding = embeddings.embed_query(query)
#     COMMON_EMBEDDINGS[query] = embedding
#     return embedding

# class AgentState(BaseModel):
#     query: str
#     session_id: str
#     conversation_history: List[Dict] = Field(default_factory=list)
#     embedding: List[float] = None
#     buckets: List[str] = Field(default_factory=list)
#     relevant_services: List[str] = Field(default_factory=list)
#     query_type: str = "saas_optimization"
#     can_answer_directly: bool = False
#     retrieval_results: Dict[str, List] = Field(default_factory=dict)
#     output: Dict[str, Any] = Field(default_factory=dict)
#     cache_key: Optional[str] = None

# class RoutingOutput(BaseModel):
#     buckets: List[str]
#     relevant_services: List[str] = Field(default_factory=list, description="List of relevant SaaS services")
#     can_answer_directly: bool = Field(default=False)
#     direct_response: str = Field(default="")
#     query_type: str = Field(default="saas_optimization", description="Type of query: basic, company_profile, saas_optimization, out_of_domain")

# class SearchItem(BaseModel):
#     title: str = Field(description="The title of the source document")
#     url: str = Field(description="The URL of the source document") 
#     category: str = Field(description="The specific category this item belongs to")

# class GenerationOutput(BaseModel):
#     use_case: List[SearchItem] = Field(default_factory=list, description="SaaS optimization services")
#     case_study: List[SearchItem] = Field(default_factory=list, description="SaaS success stories")
#     insights: List[SearchItem] = Field(default_factory=list, description="SaaS optimization insights")
#     message: str = Field(description="Focused message on relevant SaaS services")

# # Helper function (unchanged)
# def scored_vector_to_dict(scored_vector):
#     return {
#         "id": scored_vector.id,
#         "score": float(scored_vector.score),
#         "values": list(scored_vector.values) if scored_vector.values else [],
#         "metadata": dict(scored_vector.metadata) if scored_vector.metadata else {}
#     }

# COMPANY_CONTEXT = """
# THE GOOD - COMPANY OVERVIEW:
# The Good is a specialized digital experience optimization (DXO) consultancy that helps SaaS companies improve their online performance and grow revenue. We focus exclusively on conversion rate optimization (CRO) for SaaS products, not as a "do everything" agency.

# KEY SERVICES FOR SAAS:
# • Increase Registration - Optimize sign-up flows and landing pages
# • Improve Onboarding - Enhance user activation and first-time experience
# • Monetize Free Users - Convert freemium users to paid plans
# • Improve Retention - Reduce churn and increase user engagement
# • Increase Referrals - Build viral growth and referral systems
# • Mitigate Cancellations - Prevent churn through optimization

# SPECIALIZATIONS:
# • SaaS conversion optimization across the entire funnel
# • Making products "cancel-proof" through retention strategies
# • User experience enhancement for SaaS platforms
# • Data-driven growth optimization for SaaS businesses

# MISSION & VALUES:
# "Remove all the bad digital experiences until only the good remain" - We focus on ethical, sustainable growth through data-driven SaaS optimization. Our tagline is "Optimize for Good."

# TYPICAL CLIENTS:
# SaaS companies from startups to enterprise level. We've worked with major SaaS brands to increase conversion rates by 20-50% through targeted optimizations across registration, onboarding, monetization, retention, referrals, and cancellation prevention.

# 10+ years of CRO research and strategy expertise with proven SaaS optimization methodologies.
# """

# # Modified routing prompt with SaaS focus and service-centric approach
# ROUTING_PROMPT = f"""You are an AI assistant for The Good, a SaaS conversion optimization consultancy.

# COMPANY CONTEXT: {COMPANY_CONTEXT}

# Query: "{{query}}"
# History: {{conversation_history}}

# TASK: Analyze the query and determine routing strategy for SaaS optimization context.

# SAAS SERVICES TO CONSIDER:
# 1. Increase Registration - sign-up, landing pages, conversion funnels
# 2. Improve Onboarding - activation, first experience, user guidance
# 3. Monetize Free Users - freemium conversion, upgrade flows, pricing
# 4. Improve Retention - engagement, feature adoption, user success
# 5. Increase Referrals - viral growth, sharing, referral programs
# 6. Mitigate Cancellations - churn prevention, retention strategies

# ROUTING RULES:

# **BASIC QUERIES** (Greetings, simple company info):
# - "Hi", "Hello", "What is The Good?", "Who are you?"
# - Set: can_answer_directly: true, buckets: [], relevant_services: []

# **COMPANY PROFILE QUERIES** (What company does, general capabilities):
# - "What do you do?", "Tell me about your services", "How can you help?"
# - Set: can_answer_directly: false, buckets: ["services", "case-studies", "insights"], relevant_services: ["all"]

# **SPECIFIC SAAS OPTIMIZATION QUERIES** (Intent-based routing):
# - Analyze query intent and map to relevant SaaS services
# - Always include "services" bucket
# - Optionally include "case-studies" and/or "insights" if relevant
# - Set: can_answer_directly: false, buckets: ["services", ...], relevant_services: [list of matching services]

# **OUT-OF-DOMAIN QUERIES** (Non-SaaS, unrelated topics):
# - Queries about non-SaaS topics, competitors, unrelated business areas
# - Set: can_answer_directly: true, buckets: [], relevant_services: [], note: "out_of_domain"

# SERVICE MAPPING EXAMPLES:
# - "improve sign-ups" → ["Increase Registration"]
# - "reduce churn" → ["Mitigate Cancellations", "Improve Retention"]
# - "convert free users" → ["Monetize Free Users"]
# - "better onboarding" → ["Improve Onboarding"]
# - "increase referrals" → ["Increase Referrals"]
# - "user retention" → ["Improve Retention"]
# - "optimize entire funnel" → ["all"]

# OUTPUT FORMAT:
# {{
#   "buckets": ["services"] or ["services", "case-studies", "insights"] or [],
#   "relevant_services": [list of specific service names] or ["all"] or [],
#   "can_answer_directly": true/false,
#   "direct_response": "response text" (only if can_answer_directly is true),
#   "query_type": "basic" | "company_profile" | "saas_optimization" | "out_of_domain"
# }}

# Respond with JSON only."""

# # Modified generation prompt with SaaS service prioritization
# GENERATION_PROMPT = f"""Generate response for The Good SaaS optimization consultancy.

# COMPANY CONTEXT: {COMPANY_CONTEXT}

# Context: {{context}}
# Query: {{query}}
# Relevant Services: {{relevant_services}}
# Query Type: {{query_type}}
# History: {{conversation_history}}

# CONTENT FILTERING RULES:

# **SERVICES BUCKET** (Relaxed scoring for service relevance):
# - If relevant_services contains specific services, prioritize those even with scores > 0.3
# - If LLM detects intent match with any service, include it regardless of score
# - Focus on SaaS optimization services that match user intent
# - Always prioritize services bucket results when available

# **CASE-STUDIES & INSIGHTS BUCKETS** (Strict scoring):
# - Only include results with score > 0.4
# - Filter for SaaS-related content only

# RESPONSE GENERATION:

# **Company Profile Queries**: Show concise overview
# - Include multiple services, case studies, and insights
# - Highlight SaaS specialization and service areas
# - Mention 20-50% conversion improvement track record

# **Specific SaaS Optimization Queries**: Focus on relevant services
# - Prioritize matching services from relevant_services list
# - Include supporting case studies and insights when available
# - Provide actionable next steps related to the specific service area

# **Service Prioritization Order**:
# 1. Exact intent match services (from relevant_services)
# 2. Related/complementary services
# 3. Supporting case studies with high relevance
# 4. Educational insights that support the service area

# OUTPUT FORMAT:
# {{
#   "use_case": [
#     {{
#       "title": "service title",
#       "url": "service url",
#       "category": "specific service area (e.g., 'Increase Registration', 'Improve Retention')"
#     }}
#   ],
#   "case_study": [
#     {{
#       "title": "case study title", 
#       "url": "case study url",
#       "category": "relevant service area"
#     }}
#   ],
#   "insights": [
#     {{
#       "title": "insight title",
#       "url": "insight url", 
#       "category": "educational topic area"
#     }}
#   ],
#   "message": "Concise, actionable message focusing on relevant SaaS optimization services. Reference specific service areas when applicable. Include clear next steps."
# }}

# IMPORTANT: 
# - Always extract title, url, and category from context metadata accurately
# - Only include URLs that start with 'https://'
# - For services, be more lenient with scoring if intent matches
# - Keep message focused on SaaS optimization and relevant service areas
# - Mention specific services from the 6 key areas when relevant"""

# # OPTIMIZATION 5: Initialize optimized models
# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
# routing_llm = ChatOpenAI(model=ROUTING_MODEL, temperature=0.1).with_structured_output(RoutingOutput)
# generation_llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0.2).with_structured_output(GenerationOutput)
# fallback_llm = ChatOpenAI(model=FALLBACK_MODEL, temperature=0.3)

# saver = MemorySaver()

# def generate_saas_fallback(query: str, relevant_services: List[str]) -> str:
#     """Generate contextual fallback for SaaS queries"""
#     if relevant_services and relevant_services != ["all"]:
#         services_text = ", ".join(relevant_services)
#         return f"I can help you with {services_text} for your SaaS business. The Good has proven experience in optimizing these areas with 20-50% conversion improvements. Would you like to see specific case studies or learn more about our approach?"
#     else:
#         return "The Good specializes in SaaS optimization across six key areas: increasing registration, improving onboarding, monetizing free users, improving retention, increasing referrals, and mitigating cancellations. Which area is most important for your SaaS business right now?"

# # OPTIMIZATION 6: Async/parallel node functions
# def input_node(state: AgentState) -> AgentState:
#     """Optimized input with caching"""
#     logger.info(f"Input node: query='{state.query[:50]}...'")
    
#     # Generate cache key
#     state.cache_key = response_cache.get_key(state.query, state.session_id)
    
#     # Check cache first
#     cached_response = response_cache.get(state.cache_key)
#     if cached_response:
#         logger.info("Cache hit - returning cached response")
#         state.output = cached_response
#         state.can_answer_directly = True
#         return state
    
#     # Generate embedding with caching
#     state.embedding = get_cached_embedding(state.query)
#     logger.info("Generated/retrieved embedding")
#     return state

# def routing_node(state: AgentState) -> AgentState:
#     """SaaS-focused routing with service prioritization"""
#     if state.can_answer_directly:  # Skip if cached
#         logger.info("Routing node: can answer directly, skipping routing")
#         return state
        
#     history_str = str(state.conversation_history[-2:]) if state.conversation_history else ""
#     logger.info("Routing node: analyzing SaaS query")
    
#     try:
#         resp = routing_llm.invoke(ROUTING_PROMPT.format(
#             query=state.query, 
#             conversation_history=history_str
#         ))
        
#         logger.info(f"Routing response: {resp.dict()}")
#         state.buckets = resp.buckets
#         state.relevant_services = resp.relevant_services
#         state.query_type = resp.query_type
#         state.can_answer_directly = resp.can_answer_directly
#         if resp.can_answer_directly and resp.direct_response:
#             state.output = {
#                 "use_case": [],
#                 "case_study": [],
#                 "insights": [],
#                 "message": resp.direct_response
#             }
        
#         logger.info(f"Routing: buckets={state.buckets}, services={state.relevant_services}, type={state.query_type}")
        
#     except Exception as e:
#         logger.error(f"Routing error: {e}")
#         # Fallback routing - default to services
#         state.buckets = ["services"]
#         state.relevant_services = []
#         state.query_type = "saas_optimization"
#         state.can_answer_directly = False
    
#     return state

# def retrieval_node(state: AgentState) -> AgentState:
#     """Optimized parallel retrieval"""
#     if state.can_answer_directly:
#         return state
    
#     if not state.buckets:
#         logger.warning("No buckets for retrieval")
#         return state
    
#     logger.info(f"Retrieval: {len(state.buckets)} buckets")
    
#     # OPTIMIZATION 7: Reduced top_k and parallel execution
#     def retrieve_from_bucket(bucket):
#         try:
#             res = pinecone_db.query(bucket, state.embedding, top_k=3) 
#             return bucket, [scored_vector_to_dict(match) for match in res["matches"]]
#         except Exception as e:
#             logger.error(f"Retrieval error for {bucket}: {e}")
#             return bucket, []
    
#     # Parallel retrieval with timeout
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         futures = {executor.submit(retrieve_from_bucket, bucket): bucket for bucket in state.buckets}
        
#         for future in futures:
#             try:
#                 bucket, matches = future.result(timeout=5)  # 5 second timeout
#                 state.retrieval_results[bucket] = matches
#             except Exception as e:
#                 logger.error(f"Retrieval timeout/error: {e}")
#                 bucket = futures[future]
#                 state.retrieval_results[bucket] = []
    
#     return state

# def generation_node(state: AgentState) -> AgentState:
#     """SaaS-focused generation with flexible service scoring"""
#     if state.can_answer_directly and state.output.get("message"):
#         return state
    
#     # Enhanced context filtering for SaaS services
#     context = {}
#     for bucket, matches in state.retrieval_results.items():
#         if bucket == "services":
#             # Relaxed scoring for services - include lower scores if intent matches
#             service_matches = []
#             for match in matches:
#                 score = match.get("score", 0)
#                 # Include if score > 0.3 OR if it matches relevant services
#                 if score > 0.3:
#                     service_matches.append(match)
#                 elif state.relevant_services and any(
#                     service.lower() in str(match.get("metadata", {})).lower() 
#                     for service in state.relevant_services if service != "all"
#                 ):
#                     # Include even lower scores if service intent matches
#                     service_matches.append(match)
#             context[bucket] = service_matches
#         else:
#             # Strict scoring for case-studies and insights
#             high_conf = [m for m in matches if m.get("score", 0) > 0.4]
#             if high_conf:
#                 context[bucket] = high_conf
    
#     history_str = str(state.conversation_history[-2:]) if state.conversation_history else ""
    
#     try:
#         if context:
#             resp = generation_llm.invoke(GENERATION_PROMPT.format(
#                 context=str(context), 
#                 relevant_services=state.relevant_services,
#                 query_type=getattr(state, 'query_type', 'saas_optimization'),
#                 query=state.query, 
#                 conversation_history=history_str
#             ))
#             state.output = resp.dict()
#             logger.info("Generated SaaS-focused response with context")
#         else:
#             # SaaS-specific fallback
#             fallback_message = generate_saas_fallback(state.query, state.relevant_services)
#             state.output = {
#                 "use_case": [],
#                 "case_study": [],
#                 "insights": [],
#                 "message": fallback_message
#             }
#             logger.info("Used SaaS-specific fallback response")
            
#     except Exception as e:
#         logger.error(f"Generation error: {e}")
#         state.output = {
#             "use_case": [],
#             "case_study": [],
#             "insights": [],
#             "message": "I'd be happy to help optimize your SaaS business. The Good specializes in six key areas: increasing registration, improving onboarding, monetizing free users, improving retention, increasing referrals, and mitigating cancellations. Which area would you like to explore?"
#         }
    
#     return state

# def output_node(state: AgentState) -> AgentState:
#     """Optimized output with caching"""
#     logger.info("Output node: finalizing response")
    
#     # Cache the response for future queries
#     if state.cache_key and state.output:
#         response_cache.set(state.cache_key, state.output)
#         logger.info("Response cached")
    
#     # Streamlined conversation history update
#     if not state.can_answer_directly:  # Only update for non-cached responses
#         interaction = {
#             "query": state.query[:100],  # Truncate for memory efficiency
#             "timestamp": time.time(),
#             "query_type": state.query_type,
#             "relevant_services": state.relevant_services
#         }
        
#         state.conversation_history.append(interaction)
#         if len(state.conversation_history) > MEMORY_LIMIT:
#             state.conversation_history = state.conversation_history[-MEMORY_LIMIT:]
    
#     return state

# # OPTIMIZATION 8: Conditional routing with early exits
# def should_skip_retrieval(state: AgentState) -> str:
#     if state.can_answer_directly and state.output.get("message"):
#         return "output"
#     return "retrieval"

# def should_skip_generation(state: AgentState) -> str:
#     if state.can_answer_directly and state.output.get("message"):
#         return "output"
#     return "generation"

# # Graph definition (unchanged structure)
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

# app = graph.compile(checkpointer=saver)

# def run_agent(query: str, session_id: str) -> Dict[str, Any]:
#     """Optimized agent runner with timing"""
#     start_time = time.time()
#     logger.info(f"Running optimized SaaS agent: '{query[:50]}...'")
    
#     config = {"configurable": {"thread_id": session_id}}
    
#     try:
#         state = AgentState(query=query, session_id=session_id)
#         result = app.invoke(state, config=config)
        
#         duration = time.time() - start_time
#         logger.info(f"Agent completed in {duration:.2f}s")
        
#         return result['output']

#     except Exception as e:
#         duration = time.time() - start_time
#         logger.exception(f"Agent failed in {duration:.2f}s: {str(e)}")
        
#         return {
#             "use_case": [],
#             "case_study": [],
#             "insights": [],
#             "message": "I apologize, but I'm experiencing technical difficulties. Please try again later or contact The Good directly for assistance with your SaaS optimization needs."
#         }

# # OPTIMIZATION 9: Warmup function for production
# def warmup_agent():
#     """Warm up the agent with common SaaS queries"""
#     common_queries = [
#         "What is The Good?",
#         "How can you help improve conversion rates?",
#         "Help me reduce churn",
#         "Improve my onboarding process",
#         "Convert more free users to paid",
#         "Increase registration rates",
#         "What services do you offer?"
#     ]
    
#     logger.info("Warming up SaaS agent...")
#     for query in common_queries:
#         try:
#             run_agent(query, "warmup_session")
#             logger.info(f"Warmed up: {query}")
#         except Exception as e:
#             logger.error(f"Warmup failed for '{query}': {e}")
    
#     logger.info("SaaS agent warmup completed")

# # Call warmup in production
# if __name__ == "__main__":
#     warmup_agent()



"""
Optimized LangGraph agent with reduced latency and SaaS-focused routing through:
1. Parallel processing
2. Faster models for routing
3. Caching strategies
4. Optimized embedding model
5. Reduced API calls
6. SaaS service-centric approach
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
# ROUTING_MODEL = "gpt-3.5-turbo"  # Faster for simple routing
ROUTING_MODEL = "gpt-4o-mini"  # Faster for simple routing
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
    relevant_services: List[str] = Field(default_factory=list)
    query_type: str = "saas_optimization"
    can_answer_directly: bool = False
    retrieval_results: Dict[str, List] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    cache_key: Optional[str] = None

class RoutingOutput(BaseModel):
    buckets: List[str]
    relevant_services: List[str] = Field(default_factory=list, description="List of relevant SaaS services")
    can_answer_directly: bool = Field(default=False)
    direct_response: str = Field(default="")
    query_type: str = Field(default="saas_optimization", description="Type of query: basic, company_profile, saas_optimization, out_of_domain")

class SearchItem(BaseModel):
    title: str = Field(description="The title of the source document")
    url: str = Field(description="The URL of the source document") 
    category: str = Field(description="The specific category this item belongs to")

class GenerationOutput(BaseModel):
    use_case: List[SearchItem] = Field(default_factory=list, description="SaaS optimization services")
    case_study: List[SearchItem] = Field(default_factory=list, description="SaaS success stories")
    insights: List[SearchItem] = Field(default_factory=list, description="SaaS optimization insights")
    message: str = Field(description="Focused message on relevant SaaS services")

# Helper function (unchanged)
def scored_vector_to_dict(scored_vector):
    return {
        "id": scored_vector.id,
        "score": float(scored_vector.score),
        "values": list(scored_vector.values) if scored_vector.values else [],
        "metadata": dict(scored_vector.metadata) if scored_vector.metadata else {}
    }

# COMPANY_CONTEXT = """
# THE GOOD - COMPANY OVERVIEW:
# The Good is a specialized digital experience optimization (DXO) consultancy that helps SaaS companies improve their online performance and grow revenue. We focus exclusively on conversion rate optimization (CRO) for SaaS products, not as a "do everything" agency.

# KEY SERVICES FOR SAAS:
# • Increase Registration - Optimize sign-up flows and landing pages
# • Improve Onboarding - Enhance user activation and first-time experience
# • Monetize Free Users - Convert freemium users to paid plans
# • Improve Retention - Reduce churn and increase user engagement
# • Increase Referrals - Build viral growth and referral systems
# • Mitigate Cancellations - Prevent churn through optimization

# SPECIALIZATIONS:
# • SaaS conversion optimization across the entire funnel
# • Making products "cancel-proof" through retention strategies
# • User experience enhancement for SaaS platforms
# • Data-driven growth optimization for SaaS businesses

# MISSION & VALUES:
# "Remove all the bad digital experiences until only the good remain" - We focus on ethical, sustainable growth through data-driven SaaS optimization. Our tagline is "Optimize for Good."

# TYPICAL CLIENTS:
# SaaS companies from startups to enterprise level. We've worked with major SaaS brands to increase conversion rates by 20-50% through targeted optimizations across registration, onboarding, monetization, retention, referrals, and cancellation prevention.

# 10+ years of CRO research and strategy expertise with proven SaaS optimization methodologies.
# """

# # Modified routing prompt with SaaS focus and service-centric approach
# ROUTING_PROMPT = f"""You are an AI assistant for The Good, a SaaS conversion optimization consultancy.

# COMPANY CONTEXT: {COMPANY_CONTEXT}

# Query: "{{query}}"
# History: {{conversation_history}}

# TASK: Analyze the query and determine routing strategy for SaaS optimization context.

# SAAS SERVICES TO CONSIDER:
# 1. Increase Registration - sign-up, landing pages, conversion funnels
# 2. Improve Onboarding - activation, first experience, user guidance  
# 3. Monetize Free Users - freemium conversion, upgrade flows, pricing
# 4. Improve Retention - engagement, feature adoption, user success
# 5. Increase Referrals - viral growth, sharing, referral programs
# 6. Mitigate Cancellations - churn prevention, retention strategies

# ROUTING LOGIC:

# BASIC QUERIES (Greetings, simple company info):
# - "Hi", "Hello", "What is The Good?", "Who are you?"
# - Response: can_answer_directly=true, buckets=[], relevant_services=[], direct_response: "response text" (only if can_answer_directly is true, ensure to include a friendly greeting and brief company overview),

# COMPANY PROFILE QUERIES (What company does, general capabilities):
# - "What do you do?", "Tell me about your services", "How can you help?"
# - Response: can_answer_directly=false, buckets=["services", "case-studies", "insights"], relevant_services=["all"]

# SPECIFIC SAAS OPTIMIZATION QUERIES (Intent-based routing):
# - Analyze query intent and map to relevant SaaS services
# - Always include "services" bucket
# - include "case-studies" and/or "insights" if relevant
# - Response: can_answer_directly=false, buckets=["services", "case-studies", "insights"], relevant_services=[matching services]

# OUT-OF-DOMAIN QUERIES (Non-SaaS, unrelated topics):
# - Response: can_answer_directly=true, buckets=[], relevant_services=[]

# SERVICE MAPPING:
# - "improve sign-ups" → ["Increase Registration"]
# - "reduce churn" → ["Mitigate Cancellations", "Improve Retention"]  
# - "convert free users" → ["Monetize Free Users"]
# - "better onboarding" → ["Improve Onboarding"]
# - "increase referrals" → ["Increase Referrals"]
# - "user retention" → ["Improve Retention"]
# - "improve user experience" → ["Improve Onboarding", "Improve Retention"]
# - "optimize entire funnel" → ["all"]

# You must respond with valid JSON only. No additional text or explanation."""

# # Modified generation prompt with SaaS service prioritization
# GENERATION_PROMPT = f"""Generate response for The Good SaaS optimization consultancy.

# COMPANY CONTEXT: {COMPANY_CONTEXT}

# Context: {{context}}
# Query: {{query}}
# Relevant Services: {{relevant_services}}
# Query Type: {{query_type}}
# History: {{conversation_history}}

# RULES:
# 1. SERVICES BUCKET: Include results with score > 0.3 OR if intent matches relevant_services
# 2. CASE-STUDIES & INSIGHTS: Only include results with score > 0.4
# 3. Extract title, url, and category from context metadata accurately **only if url starts with 'https://', is unique, and among duplicates keep the one with the highest similarity score**
# 4. Focus on SaaS optimization and relevant service areas
# 5. Do not include same URLs 

# For Company Profile Queries: Show comprehensive overview with multiple services
# For Specific SaaS Queries: Focus on matching services from relevant_services list

# You must respond with valid JSON only. No additional text or explanation."""

COMPANY_CONTEXT = """
THE GOOD - COMPANY OVERVIEW:
The Good is a specialized digital experience optimization (DXO) consultancy that helps SaaS companies improve their online performance and grow revenue. We focus exclusively on conversion rate optimization (CRO) for SaaS products, not as a "do everything" agency.

KEY SERVICES FOR SAAS:
• Increase Registration - Optimize sign-up flows and landing pages
• Improve Onboarding - Enhance user activation and first-time experience
• Monetize Free Users - Convert freemium users to paid plans
• Improve Retention - Reduce churn and increase user engagement
• Increase Referrals - Build viral growth and referral systems
• Mitigate Cancellations - Prevent churn through optimization

SPECIALIZATIONS:
• SaaS conversion optimization across the entire funnel
• Making products "cancel-proof" through retention strategies
• User experience enhancement for SaaS platforms
• Data-driven growth optimization for SaaS businesses

MISSION & VALUES:
"Remove all the bad digital experiences until only the good remain" - We focus on ethical, sustainable growth through data-driven SaaS optimization. Our tagline is "Optimize for Good."

TYPICAL CLIENTS:
SaaS companies from startups to enterprise level. We've worked with major SaaS brands to increase conversion rates by 20-50% through targeted optimizations across registration, onboarding, monetization, retention, referrals, and cancellation prevention.

10+ years of CRO research and strategy expertise with proven SaaS optimization methodologies.
"""

# Improved routing prompt with intent-based service mapping
ROUTING_PROMPT = f"""You are an AI assistant for The Good, a SaaS conversion optimization consultancy.

COMPANY CONTEXT: {COMPANY_CONTEXT}

Query: "{{query}}"
History: {{conversation_history}}

TASK: Analyze the query intent and determine appropriate routing strategy for SaaS optimization context.

AVAILABLE SAAS SERVICES:
1. Increase Registration - sign-up optimization, landing page conversion, lead generation
2. Improve Onboarding - user activation, first-time experience, product adoption
3. Monetize Free Users - freemium to paid conversion, upgrade optimization, pricing strategy
4. Improve Retention - user engagement, feature adoption, customer success, loyalty
5. Increase Referrals - viral growth mechanisms, sharing features, referral programs
6. Mitigate Cancellations - churn prevention, retention strategies, win-back campaigns

INTENT ANALYSIS APPROACH:

QUERY CLASSIFICATION:
1. BASIC QUERIES: Simple greetings or company identification
   - Response: can_answer_directly=true, buckets=[], relevant_services=[], direct_response: "response text"

2. COMPANY PROFILE QUERIES: General company capabilities or broad service inquiries
   - Response: can_answer_directly=false, buckets=["services", "case-studies", "insights"], relevant_services=["all"]

3. SPECIFIC SAAS OPTIMIZATION QUERIES: Any query expressing a business challenge, goal, or problem
   - Response: can_answer_directly=false, buckets=["services", "case-studies", "insights"], relevant_services=[determined by analysis]

4. OUT-OF-DOMAIN QUERIES: Non-SaaS, unrelated topics
   - Response: can_answer_directly=true, buckets=[], relevant_services=[]

DYNAMIC SERVICE MAPPING INSTRUCTIONS:
For SPECIFIC SAAS OPTIMIZATION QUERIES, analyze the user's query to understand their underlying business intent and challenges. Then determine which of our services would be most relevant:

AVAILABLE SERVICES:
• "Increase Registration" - Helps with sign-up flows, landing pages, and user acquisition
• "Improve Onboarding" - Enhances user activation and first-time experience  
• "Monetize Free Users" - Converts freemium users to paid plans
• "Improve Retention" - Reduces churn and increases user engagement
• "Increase Referrals" - Builds viral growth and referral systems
• "Mitigate Cancellations" - Prevents churn through optimization

ANALYSIS PROCESS:
1. Extract the core business problem, challenge, or goal from the user's query
2. Consider what stage of the SaaS customer journey this relates to
3. Think about which service(s) would directly address their stated or implied needs
4. Select relevant services based on logical connections to their business challenge
5. If the query is broad or mentions multiple aspects, include multiple relevant services
6. If unclear or very comprehensive, use ["all"]

Your job is to understand the intent behind any query, regardless of how it's phrased, and intelligently map it to the most appropriate service(s) that would help solve their business challenge.

You must respond with valid JSON only. No additional text or explanation."""

# Enhanced generation prompt with better service prioritization
GENERATION_PROMPT = f"""Generate response for The Good SaaS optimization consultancy.

COMPANY CONTEXT: {COMPANY_CONTEXT}

Context: {{context}}
Query: {{query}}
Relevant Services: {{relevant_services}}
Query Type: {{query_type}}
History: {{conversation_history}}

RESPONSE GENERATION RULES:
1. SERVICES BUCKET: Include results with score > 0.3 OR if content relates to relevant_services
2. CASE-STUDIES & INSIGHTS: Only include results with score > 0.4  
3. URL VALIDATION: Only include URLs that start with 'https://' and are unique
4. DUPLICATE HANDLING: Among duplicate URLs, keep only the one with highest similarity score
5. SERVICE PRIORITIZATION: Prioritize content that matches the identified relevant_services
6. METADATA EXTRACTION: Extract title, url, and category accurately from context metadata

RESPONSE STRATEGY BY QUERY TYPE:
- Company Profile Queries: Provide comprehensive overview highlighting multiple relevant services
- Specific SaaS Queries: Focus primarily on matching services from relevant_services list, with supporting case studies and insights
- Multi-service Queries: Balance coverage across all relevant service areas

CONTENT ORGANIZATION:
- Lead with most relevant service information
- Support with applicable case studies showing results
- Include insights that demonstrate expertise in the identified service areas
- Maintain focus on SaaS optimization throughout

You must respond with valid JSON only. No additional text or explanation."""

# OPTIMIZATION 5: Initialize optimized models
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
routing_llm = ChatOpenAI(model=ROUTING_MODEL, temperature=0.1).with_structured_output(RoutingOutput)
generation_llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0.2).with_structured_output(GenerationOutput)
fallback_llm = ChatOpenAI(model=FALLBACK_MODEL, temperature=0.3)

saver = MemorySaver()

def generate_saas_fallback(query: str, relevant_services: List[str]) -> str:
    """Generate contextual fallback for SaaS queries"""
    if relevant_services and relevant_services != ["all"]:
        services_text = ", ".join(relevant_services)
        return f"I can help you with {services_text} for your SaaS business. The Good has proven experience in optimizing these areas with 20-50% conversion improvements. Would you like to see specific case studies or learn more about our approach?"
    else:
        return "The Good specializes in SaaS optimization across six key areas: increasing registration, improving onboarding, monetizing free users, improving retention, increasing referrals, and mitigating cancellations. Which area is most important for your SaaS business right now?"

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
    """SaaS-focused routing with service prioritization"""
    if state.can_answer_directly:  # Skip if cached
        return state
        
    history_str = str(state.conversation_history[-2:]) if state.conversation_history else ""
    logger.info("Routing node: analyzing SaaS query")
    
    try:
        # Add retry logic for routing
        max_retries = 2
        for attempt in range(max_retries):
            try:
                resp = routing_llm.invoke(ROUTING_PROMPT.format(
                    query=state.query, 
                    conversation_history=history_str
                ))
                
                state.buckets = resp.buckets
                state.relevant_services = resp.relevant_services
                state.query_type = resp.query_type
                state.can_answer_directly = resp.can_answer_directly
                
                if resp.can_answer_directly and resp.direct_response:
                    state.output = {
                        "use_case": [],
                        "case_study": [],
                        "insights": [],
                        "message": resp.direct_response
                    }
                
                logger.info(f"Routing: buckets={state.buckets}, services={state.relevant_services}, type={state.query_type}")
                break
                
            except Exception as retry_error:
                logger.warning(f"Routing attempt {attempt + 1} failed: {retry_error}")
                if attempt == max_retries - 1:
                    raise retry_error
        
    except Exception as e:
        logger.error(f"Routing error after retries: {e}")
        # Enhanced fallback routing for UX-related queries
        query_lower = state.query.lower()
        if any(term in query_lower for term in ["user experience", "ux", "improve experience", "user journey"]):
            state.buckets = ["services"]
            state.relevant_services = ["Improve Onboarding", "Improve Retention"]
            state.query_type = "saas_optimization" 
        else:
            state.buckets = ["services"]
            state.relevant_services = []
            state.query_type = "saas_optimization"
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
    """SaaS-focused generation with flexible service scoring"""
    if state.can_answer_directly and state.output.get("message"):
        return state
    
    # Enhanced context filtering for SaaS services
    context = {}
    for bucket, matches in state.retrieval_results.items():
        if bucket == "services":
            # Relaxed scoring for services - include lower scores if intent matches
            service_matches = []
            for match in matches:
                score = match.get("score", 0)
                # Include if score > 0.3 OR if it matches relevant services
                if score > 0.3:
                    service_matches.append(match)
                elif state.relevant_services and any(
                    service.lower() in str(match.get("metadata", {})).lower() 
                    for service in state.relevant_services if service != "all"
                ):
                    # Include even lower scores if service intent matches
                    service_matches.append(match)
            context[bucket] = service_matches
        else:
            # Strict scoring for case-studies and insights
            high_conf = [m for m in matches if m.get("score", 0) > 0.35]
            if high_conf:
                context[bucket] = high_conf
    
    history_str = str(state.conversation_history[-2:]) if state.conversation_history else ""
    
    try:
        if context:
            # Add retry logic for generation
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    resp = generation_llm.invoke(GENERATION_PROMPT.format(
                        context=str(context), 
                        relevant_services=state.relevant_services,
                        query_type=getattr(state, 'query_type', 'saas_optimization'),
                        query=state.query, 
                        conversation_history=history_str
                    ))
                    state.output = resp.dict()
                    logger.info("Generated SaaS-focused response with context")
                    break
                    
                except Exception as retry_error:
                    logger.warning(f"Generation attempt {attempt + 1} failed: {retry_error}")
                    if attempt == max_retries - 1:
                        raise retry_error
        else:
            # SaaS-specific fallback
            fallback_message = generate_saas_fallback(state.query, state.relevant_services)
            state.output = {
                "use_case": [],
                "case_study": [],
                "insights": [],
                "message": fallback_message
            }
            logger.info("Used SaaS-specific fallback response")
            
    except Exception as e:
        logger.error(f"Generation error after retries: {e}")
        state.output = {
            "use_case": [],
            "case_study": [],
            "insights": [],
            "message": "I'd be happy to help optimize your SaaS business. The Good specializes in six key areas: increasing registration, improving onboarding, monetizing free users, improving retention, increasing referrals, and mitigating cancellations. Which area would you like to explore?"
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
            "query_type": state.query_type,
            "relevant_services": state.relevant_services
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
    logger.info(f"Running optimized SaaS agent: '{query[:50]}...'")
    
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
            "message": "I apologize, but I'm experiencing technical difficulties. Please try again later or contact The Good directly for assistance with your SaaS optimization needs."
        }

# OPTIMIZATION 9: Warmup function for production
def warmup_agent():
    """Warm up the agent with common SaaS queries"""
    common_queries = [
        "What is The Good?",
        "How can you help improve conversion rates?",
        "Help me reduce churn",
        "Improve my onboarding process",
        "Convert more free users to paid",
        "Increase registration rates",
        "What services do you offer?"
    ]
    
    logger.info("Warming up SaaS agent...")
    for query in common_queries:
        try:
            run_agent(query, "warmup_session")
            logger.info(f"Warmed up: {query}")
        except Exception as e:
            logger.error(f"Warmup failed for '{query}': {e}")
    
    logger.info("SaaS agent warmup completed")

# Call warmup in production
if __name__ == "__main__":
    warmup_agent()