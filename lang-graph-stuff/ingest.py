# # ingest.py
# """
# Script to ingest data from JSON files into Pinecone indexes.
# Usage: python ingest.py
# """
# import json
# import os
# from db import pinecone_db
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings

# load_dotenv()

# DATA_FILES = {
#     "services": "services.json",
#     "case-studies": "case-studies.json",
#     "insights": "insights.json"
# }

# EMBEDDING_MODEL = "text-embedding-3-large"

# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# def load_json(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def ingest():
#     for bucket, fname in DATA_FILES.items():
#         if not os.path.exists(fname):
#             print(f"File {fname} not found, skipping {bucket}.")
#             continue
#         data = load_json(fname)
#         vectors = []
#         for i, item in enumerate(data):
#             text = item.get("text") or item.get("content") or str(item)
#             meta = {k: v for k, v in item.items() if k != "text" and k != "content"}
#             emb = embeddings.embed_query(text)
#             vectors.append({
#                 "id": f"{bucket}-{i}",
#                 "values": emb,
#                 "metadata": meta
#             })
#         if vectors:
#             pinecone_db.upsert(bucket, vectors)
#             print(f"Ingested {len(vectors)} items into {bucket}.")

# if __name__ == "__main__":
#     ingest()


# ingest.py


"""
Script to ingest data from a structured JSON file into Pinecone indexes.
The JSON is expected to be from the extraction script, with fields like:
- title, category (use_case/case_study/insight), use_case, date, chunk_index, text_chunk, url

Usage: python ingest.py
"""
import json
import os
from db import pinecone_db
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import re  # For simple slug generation

load_dotenv()

# Configurable input file (from extraction script)
DATA_FILE = "structured_data_latest.json"

# Mapping from extraction categories to Pinecone buckets
CATEGORY_TO_BUCKET = {
    "use_case": "services",
    "case_study": "case-studies",
    "insight": "insights"
}

EMBEDDING_MODEL = "text-embedding-3-large"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_slug(title: str) -> str:
    """Generate a simple slug from title for unique IDs."""
    return re.sub(r'\W+', '-', title.lower()).strip('-')[:50]  # Limit length

def ingest():
    if not os.path.exists(DATA_FILE):
        print(f"File {DATA_FILE} not found. Exiting.")
        return

    data = load_json(DATA_FILE)

    # Group data by bucket
    grouped_data = {bucket: [] for bucket in CATEGORY_TO_BUCKET.values()}

    for item in data:
        category = item.get("category")
        bucket = CATEGORY_TO_BUCKET.get(category)
        if not bucket:
            print(f"Skipping item with unknown category '{category}'.")
            continue
        grouped_data[bucket].append(item)

    # Ingest per bucket
    for bucket, items in grouped_data.items():
        if not items:
            print(f"No items for {bucket}, skipping.")
            continue

        vectors = []
        for item in items:
            text = item.get("text_chunk")
            if not text:
                print(f"Skipping item in {bucket} with no text_chunk.")
                continue

            # Metadata: Exclude text_chunk to avoid redundancy
            meta = {
                "title": item.get("title", ""),
                "use_case": item.get("use_case", ""),
                "date": item.get("date", None),
                "chunk_index": item.get("chunk_index", 0),
                "url": item.get("url", ""),
                "category": item.get("category", "")
            }

            # Embed the text_chunk
            emb = embeddings.embed_query(text)

            # Unique ID
            title_slug = generate_slug(meta["title"])
            id = f"{bucket}-{title_slug}-{meta['chunk_index']}"

            vectors.append({
                "id": id,
                "values": emb,
                "metadata": meta
            })

        if vectors:
            pinecone_db.upsert(bucket, vectors)
            print(f"Ingested {len(vectors)} items into {bucket}.")

if __name__ == "__main__":
    ingest()
