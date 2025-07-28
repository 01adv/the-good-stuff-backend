do everything inside this folder: lang-graph-stuff, do not litter anything outside of this,

Build a complete backend for a SaaS optimization project using FastAPI, LangGraph, Pinecone, and OpenAI. The backend should orchestrate an AI agent that handles user queries by semantically searching across three separate vector database buckets: 'services' (our company's specializations and differentiations mapped to customer funnel stages like onboarding, registration, usage, trial, paid, renewal, referral, cancellation), 'case-studies' (real-world examples), and 'insights' (articles and metrics). The agent decides which buckets to query based on query intent, retrieves results using RAG, maps to our services only if confidence (cosine similarity > 0.8) is high, and handles no-results cases by acknowledging limitations (e.g., for 'payment fraud', respond with 'No direct match, but explore our related onboarding services').

Key requirements:

- Use FastAPI for async API endpoints, including a /search POST endpoint that takes a JSON body with 'query' (string) and returns JSON with: use_case (string), case_study (string), insights (list of 3 strings), and message (helpful string).
- Use LangGraph to build a stateful graph for the agent with the following nodes and edges:
  - Input Node: Receive and embed the query using OpenAI text-embedding-3-large.
  - Routing Node: Use OpenAI GPT-4o LLM to analyze query intent and decide which buckets to query (e.g., 'increase users' routes to services and case-studies; 'payment fraud' to insights or none). Maintain state with funnel context.
  - Retrieval Nodes: Parallel retrieval from selected Pinecone buckets using cosine similarity (top-k=3). Aggregate results.
  - Generation Node: Use GPT-4o to process retrieved context, check confidence, map to services if applicable, and generate the output JSON. If no results or low confidence, generate an acknowledgment message.
  - Output Node: Format and return the JSON.
  - Incorporate session-based memory using LangGraph's CheckpointSaver with an in-memory store (e.g., SqliteSaver for testing) tied to a session_id in the API request. Persist only recent query context (e.g., last 3 interactions) for short periods, expiring after 10 minutes. Use this to enhance routing and generation nodes with prior context if relevant
  - Edges: Conditional branching (e.g., if no buckets selected, edge directly to generation for no-results handling; use state to persist confidence scores).
- Vector DB Setup: Use Pinecone with three separate indexes ('services-index', 'case-studies-index', 'insights-index'), each with 1536 dimensions. Include a script to ingest sample data (assume JSON files for scraped data: services.json, case-studies.json, insights.json) by generating embeddings and upserting with metadata (e.g., {'funnel_stage': 'onboarding'}).
- LLMs and Embeddings: Integrate OpenAI via langchain-openai. Use GPT-4o for reasoning and text-embedding-3-large for embeddings. Include prompts for routing and generation (e.g., "Analyze query: {query}. Route to buckets: services, case-studies, insights. Map to funnel stages if similarity > 0.8").
- Other Tools: Integrate Langfuse for monitoring (trace agent runs). Use Pydantic for data models. Add error handling and logging.
- Project Structure: Organize as a Python project with main.py (FastAPI app), agent.py (LangGraph graph), db.py (Pinecone utils), ingest.py (data ingestion script), requirements.txt, and Dockerfile for deployment.
- Environment: Use .env for keys (OPENAI_API_KEY, PINECONE_API_KEY, LANGFUSE_SECRET_KEY). Assume Python 3.11.
- Testing: Include sample queries in comments (e.g., 'increase users' should return onboarding-related results; 'payment fraud' should acknowledge no match).
- Best Practices: Ensure scalability, async operations, and no chatbot-like behavior—just render results. Do not include frontend code.

Generate the full code, including all files, with explanations in comments. Make it production-ready and easy to run with 'uvicorn main:app --reload'.
