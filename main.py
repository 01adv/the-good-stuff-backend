import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai
from rag_stuff.vector_search import vector_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not found. OpenAI fallback will not work.")
    client = None

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

logger.info("Ready to serve vector search queries.")


@app.post("/ask")
async def ask_query(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    try:
        results = vector_search(request.query, top_k=10)
        logger.info(f"Vector search results: {results}")
        # results = [r for r in results if abs(r["score"]) > 0.75]

        if not results:
            logger.info("No results found for query. Using fallback RAG reasoning.")
            answer = "No relevant matches found."
            if client:
                # Fallback: Use OpenAI to generate a helpful answer
                prompt = f"The user asked: '{request.query}'\n\nWe couldn't find relevant matches. Please provide a helpful answer based on your SaaS knowledge."
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful SaaS assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    answer = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"OpenAI fallback failed: {e}")
            return {
                "answer": answer,
                "clusters": []
            }

        # Build clusters by subcategory
        clustered_sources = {}
        for hit in results:
            meta = hit["metadata"]
            subcat = meta.get("subcategory", "Uncategorized")
            if subcat not in clustered_sources:
                clustered_sources[subcat] = []
            clustered_sources[subcat].append({
                "title": meta.get("title"),
                "url": meta.get("url"),
                "content_type": meta.get("content_type"),
                "score": hit.get("score"),
                "subcategory": meta.get("subcategory")
            })

        # Compose answer from top result
        answer = results[0]["document"] if results else "No relevant matches found."
        logger.info("Returning answer and clusters.")
        return {
            "answer": answer,
            "clusters": clustered_sources
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
