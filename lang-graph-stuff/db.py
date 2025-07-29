# db.py
# """
# Pinecone DB utilities for vector search and ingestion.
# Handles three indexes: services, case-studies, insights.
# """
# import os
# from typing import List, Dict, Any
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv

# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")

# INDEXES = {
#     "services": "services-index",
#     "case-studies": "case-studies-index",
#     "insights": "insights-index"
# }

# DIMENSIONS = 1536

# class PineconeDB:
#     def __init__(self):
#         self.pc = Pinecone(api_key=PINECONE_API_KEY)
#         self.env = PINECONE_ENV
#         self.indexes = {}
#         for k, v in INDEXES.items():
#             if v not in self.pc.list_indexes().names():
#                 self.pc.create_index(name=v, dimension=DIMENSIONS, spec=ServerlessSpec(cloud="aws", region=self.env))
#             self.indexes[k] = self.pc.Index(v)

#     def upsert(self, bucket: str, vectors: List[Dict[str, Any]]):
#         """Upsert vectors to the specified bucket/index."""
#         idx = self.indexes[bucket]
#         idx.upsert(vectors)

#     def query(self, bucket: str, embedding: List[float], top_k: int = 3):
#         idx = self.indexes[bucket]
#         return idx.query(vector=embedding, top_k=top_k, include_metadata=True)

# # Singleton for app
# pinecone_db = PineconeDB()



# db.py
"""
Pinecone DB utilities for vector search and ingestion.
Handles three indexes: services, case-studies, insights.
"""
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west-2")  # Fallback to a default region

INDEXES = {
    "services": "services-index",
    "case-studies": "case-studies-index",
    "insights": "insights-index"
}

DIMENSIONS = 3072  # Matches default for text-embedding-3-large; adjust if using reduced dimensions
METRIC = "cosine"  # Explicit for semantic similarity
BATCH_SIZE = 100  # For efficient upserts

class PineconeDB:
    def __init__(self):
        logger.info("Initializing PineconeDB...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.env = PINECONE_ENV
        self.indexes = {}
        existing_indexes = self.pc.list_indexes().names()
        logger.debug(f"Existing indexes: {existing_indexes}")
        for k, v in INDEXES.items():
            if v not in existing_indexes:
                try:
                    logger.info(f"Creating index '{v}' for bucket '{k}'...")
                    self.pc.create_index(
                        name=v,
                        dimension=DIMENSIONS,
                        metric=METRIC,
                        spec=ServerlessSpec(cloud="aws", region=self.env)
                    )
                    logger.success(f"Created index '{v}' for bucket '{k}'.")
                except Exception as e:
                    logger.error(f"Error creating index '{v}': {e}")
            else:
                logger.info(f"Index '{v}' already exists for bucket '{k}'.")
            self.indexes[k] = self.pc.Index(v)
        logger.success("PineconeDB initialization complete.")

    def upsert(self, bucket: str, vectors: List[Dict[str, Any]]):
        """Upsert vectors to the specified bucket/index in batches."""
        logger.info(f"Upserting {len(vectors)} vectors to bucket '{bucket}'...")
        idx = self.indexes.get(bucket)
        if not idx:
            logger.error(f"Unknown bucket: {bucket}")
            raise ValueError(f"Unknown bucket: {bucket}")

        total_upserted = 0
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            idx.upsert(batch)
            total_upserted += len(batch)
            logger.debug(f"Upserted batch of {len(batch)} vectors to {bucket} (total: {total_upserted}).")

        logger.success(f"Completed upsert for {bucket} with {total_upserted} vectors.")

    def query(self, bucket: str, embedding: List[float], top_k: int = 3, filter: Optional[Dict[str, Any]] = None):
        """Query the specified bucket/index with optional metadata filter."""
        logger.info(f"Querying bucket '{bucket}' with top_k={top_k} and filter={filter}...")
        idx = self.indexes.get(bucket)
        if not idx:
            logger.error(f"Unknown bucket: {bucket}")
            raise ValueError(f"Unknown bucket: {bucket}")

        query_params = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        if filter:
            query_params["filter"] = filter  # E.g., {"use_case": "increase_registration"}

        result = idx.query(**query_params)
        logger.success(f"Query completed for bucket '{bucket}'.")
        return result

# Singleton for app
pinecone_db = PineconeDB()
