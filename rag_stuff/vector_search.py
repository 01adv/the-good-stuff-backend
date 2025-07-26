import os
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
import chromadb
import numpy as np
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()


# llm
# llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# module‑level singletons
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "saas_optimization"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
embedding_fn = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def preprocess_filters(filters: dict) -> dict:
    if not filters:
        return {}

    categories = []
    price_values = []
    other_filters = {}

    for key, value in filters.items():
        if key == "category" and isinstance(value, str):
            categories = [c.strip() for c in value.split(",")]
        elif key == "price":
            if isinstance(value, dict):
                price_val = value.get("$lte")
                if isinstance(price_val, str) and "," in price_val:
                    price_values = [int(x.strip()) for x in price_val.split(",")]
                elif isinstance(price_val, (int, float)):
                    price_values = [price_val]
            elif isinstance(value, (int, float)):
                price_values = [value]
        else:
            other_filters[key] = value

    # Build OR filter blocks
    or_block = []
    if categories:
        for cat in categories:
            if price_values:
                for price in price_values:
                    clause = {"category": cat, "price": {"$lte": price}}
                    clause.update(other_filters)
                    or_block.append(clause)
            else:
                clause = {"category": cat}
                clause.update(other_filters)
                or_block.append(clause)
    elif price_values:
        for price in price_values:
            clause = {"price": {"$lte": price}}
            clause.update(other_filters)
            or_block.append(clause)

    # Simplify return if only one clause
    if len(or_block) == 1:
        return or_block[0]
    elif or_block:
        return {"$or": or_block}

    # Fallback: $and logic for other filters
    if len(filters) > 1:
        return {"$and": [{k: v} for k, v in filters.items()]}

    return filters




def normalize(vec):
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def vector_search(query: str, top_k: int = 5, filters: dict | None = None) -> list:
    logger.info(f"Collection '{COLLECTION_NAME}' initialized in vector_search")

    if isinstance(query, str) and query.lstrip().startswith("{"):
        data = json.loads(query)
        query   = data.get("query", query)
        top_k   = data.get("top_k", top_k)
        # filters = data.get("filters", filters)  # filters ignored for now

    # if filters:
    #     filters = preprocess_filters(filters)  # filters ignored for now

    # Embed and normalize query
    raw_embedding = embedding_fn([query])[0]
    query_vec = normalize(raw_embedding)

    def run(qvec):
        query_params = {
            "query_embeddings": [qvec],
            "n_results": top_k,
            "include": ["metadatas", "documents"]
        }
        logger.info(f"Querying with params: {query_params}")
        return collection.query(**query_params)

    logger.info(f"Running vector search with query: {query}")
    res = run(query_vec)

    hits = [
        {"metadata": m, "document": d, "score": 1 - x}
        for m, d, x in zip(res["metadatas"][0], res["documents"][0], res["distances"][0])
    ]
    return hits




# def vector_search(query: str, top_k: int = 5, filters: dict | None = None) -> list:
#     logger.info(f"Collection '{COLLECTION_NAME}' initialized in vector_search")

#     if isinstance(query, str) and query.lstrip().startswith("{"):
#         data = json.loads(query)
#         query   = data.get("query", query)
#         top_k   = data.get("top_k", top_k)
#         filters = data.get("filters", filters)

#     if filters:
#         filters = preprocess_filters(filters)

#     # Embed and normalize query
#     raw_embedding = embedding_fn([query])[0]
#     query_vec = normalize(raw_embedding)

#     def run(qvec, where):
#         query_params = {
#             "query_embeddings": [qvec],
#             "n_results": top_k,
#             "include": ["metadatas", "documents", "distances", "ids"]
#         }
#         if where:
#             query_params["where"] = where
#         logger.info(f"Querying with params: {query_params}")
#         return collection.query(**query_params)

#     logger.info(f"Running vector search with query: {query}, filters: {filters}")
#     res = run(query_vec, filters)

#     if not res["metadatas"][0]:
#         logger.info("No results with filters, trying without filters")
#         res = run(query_vec, None)

#     hits = [
#         {"metadata": m, "document": d, "score": 1 - x}
#         for m, d, x in zip(res["metadatas"][0], res["documents"][0], res["distances"][0])
#     ]
#     return hits
