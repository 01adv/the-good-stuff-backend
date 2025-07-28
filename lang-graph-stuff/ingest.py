# ingest.py
"""
Script to ingest data from JSON files into Pinecone indexes.
Usage: python ingest.py
"""
import json
import os
from db import pinecone_db
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_FILES = {
    "services": "services.json",
    "case-studies": "case-studies.json",
    "insights": "insights.json"
}

EMBEDDING_MODEL = "text-embedding-3-large"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ingest():
    for bucket, fname in DATA_FILES.items():
        if not os.path.exists(fname):
            print(f"File {fname} not found, skipping {bucket}.")
            continue
        data = load_json(fname)
        vectors = []
        for i, item in enumerate(data):
            text = item.get("text") or item.get("content") or str(item)
            meta = {k: v for k, v in item.items() if k != "text" and k != "content"}
            emb = embeddings.embed_query(text)
            vectors.append({
                "id": f"{bucket}-{i}",
                "values": emb,
                "metadata": meta
            })
        if vectors:
            pinecone_db.upsert(bucket, vectors)
            print(f"Ingested {len(vectors)} items into {bucket}.")

if __name__ == "__main__":
    ingest()
