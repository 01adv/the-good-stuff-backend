import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db import pinecone_db, INDEXES, DIMENSIONS, METRIC, PINECONE_API_KEY, PINECONE_ENV

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_slug(title: str) -> str:
    """Generate a URL-friendly slug from a title."""
    return re.sub(r'\W+', '-', title.lower()).strip('-')[:50]


# def replace_services_index(json_path: str = "use_case_sections.json"):
def replace_services_index(json_path: str = None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "use_case_sections.json")
    """Deletes and replaces the entire 'services' index with fresh data from JSON."""
    if not os.path.exists(json_path):
        print(f"File {json_path} not found.")
        return

    data = load_json(json_path)
    use_case_chunks = [item for item in data if item.get("category") == "use_case"]

    if not use_case_chunks:
        print("No 'use_case' entries found in data.")
        return

    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = INDEXES["services"]

    try:
        pc.delete_index(index_name)
        print(f"[✓] Deleted old index '{index_name}'.")
    except Exception as e:
        print(f"[!] Error deleting index '{index_name}': {e}")
        return

    try:
        pc.create_index(
            name=index_name,
            dimension=DIMENSIONS,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        print(f"[✓] Recreated index '{index_name}'.")
    except Exception as e:
        print(f"[!] Error creating index '{index_name}': {e}")
        return

    # Attach the new index to your PineconeDB wrapper
    pinecone_db.indexes["services"] = pc.Index(index_name)

    vectors = []
    for item in use_case_chunks:
        text = item.get("text_chunk")
        if not text:
            continue

        meta = {
            "title": item.get("title", ""),
            "use_case": item.get("use_case", ""),
            # "date": item.get("date", None),
            "chunk_index": item.get("chunk_index", 0),
            "url": item.get("url", "")
        }

        emb = embeddings.embed_query(text)
        title_slug = generate_slug(meta["title"])
        vector_id = f"services-{title_slug}-{meta['chunk_index']}"

        vectors.append({
            "id": vector_id,
            "values": emb,
            "metadata": meta
        })

    if vectors:
        pinecone_db.upsert("services", vectors)
        print(f"[✓] Replaced 'services' index with {len(vectors)} vectors.")
    else:
        print("[!] No valid vectors to upsert.")


# Optional trigger
if __name__ == "__main__":
    replace_services_index()
