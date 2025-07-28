# db.py
"""
Pinecone DB utilities for vector search and ingestion.
Handles three indexes: services, case-studies, insights.
"""
import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

INDEXES = {
    "services": "services-index",
    "case-studies": "case-studies-index",
    "insights": "insights-index"
}

DIMENSIONS = 1536

class PineconeDB:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.env = PINECONE_ENV
        self.indexes = {}
        for k, v in INDEXES.items():
            if v not in self.pc.list_indexes().names():
                self.pc.create_index(name=v, dimension=DIMENSIONS, spec=ServerlessSpec(cloud="aws", region=self.env))
            self.indexes[k] = self.pc.Index(v)

    def upsert(self, bucket: str, vectors: List[Dict[str, Any]]):
        """Upsert vectors to the specified bucket/index."""
        idx = self.indexes[bucket]
        idx.upsert(vectors)

    def query(self, bucket: str, embedding: List[float], top_k: int = 3):
        idx = self.indexes[bucket]
        return idx.query(vector=embedding, top_k=top_k, include_metadata=True)

# Singleton for app
pinecone_db = PineconeDB()
