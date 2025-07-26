# import os
# import json
# import shutil
# import numpy as np
# import logging
# from dotenv import load_dotenv
# import chromadb
# from chromadb.config import Settings
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# # ─── Logging ────────────────────────────────────────────────────────────────────
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s | %(levelname)s | %(message)s')
# logger = logging.getLogger(__name__)

# # Load env vars
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Constants
# DATA_PATH = "extraction-work/structured_data3.json"
# CHROMA_DB_DIR = "chroma_db"
# COLLECTION_NAME = "saas_optimization"

# # ─── Helper ─────────────────────────────────────────────────────────────────────
# def normalize(vec):
#     """Return unit-length vector (for cosine similarity)."""
#     v = np.asarray(vec, dtype=np.float32)
#     norm = np.linalg.norm(v)
#     return v / norm if norm > 0 else v

# # Step 1: Delete previous DB if needed
# def delete_chroma_db(path):
#     if os.path.exists(path):
#         logger.info(f"Deleting old ChromaDB at {path}")
#         shutil.rmtree(path)
#     else:
#         logger.info("No existing ChromaDB directory found.")

# delete_chroma_db(CHROMA_DB_DIR)

# # Step 2: Initialize ChromaDB client with persistence
# client = chromadb.PersistentClient(path="./chroma_db")

# # Step 3: Configure OpenAI embeddings
# embedding_fn = OpenAIEmbeddingFunction(
#     api_key=OPENAI_API_KEY,
#     model_name="text-embedding-3-small"
# )

# # Step 4: Create or reset collection
# try:
#     client.delete_collection(name=COLLECTION_NAME)
#     logger.info(f"Existing '{COLLECTION_NAME}' collection deleted.")
# except Exception:
#     logger.info(f"No existing '{COLLECTION_NAME}' collection to delete.")
# collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

# # Step 5: Load and prepare JSON data
# try:
#     with open(DATA_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     logger.info(f"Loaded {len(data)} items from {DATA_PATH}.")
# except Exception as e:
#     logger.error(f"Failed to load JSON data: {e}")
#     raise

# # Step 6: Chunk and prepare data
# def chunk_text(text, chunk_size=500, overlap=50):
#     """Split text into chunks with specified size and overlap."""
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap
#     return chunks

# ids, documents, metadatas, raw_embeddings = [], [], [], []

# for item in data:
#     required_keys = ["title", "text", "content_description", "subcategory", "content_type", "url", "category"]
#     if not all(k in item for k in required_keys):
#         logger.warning(f"Skipping item missing required keys: {item.get('url', 'unknown')}")
#         continue

#     text = f"{item['title']} {item['text']} {item['content_description']}".strip()
#     if not text:
#         logger.warning(f"Skipping empty text for item: {item.get('url', 'unknown')}")
#         continue

#     chunks = chunk_text(text)
#     for i, chunk in enumerate(chunks):
#         doc_id = f"{item['url']}_{i}"
#         metadata = {
#             "title": item["title"],
#             "url": item["url"],
#             "category": item["category"],
#             "subcategory": item["subcategory"],
#             "content_type": item["content_type"],
#             "chunk_index": i
#         }
#         ids.append(doc_id)
#         documents.append(chunk)
#         metadatas.append(metadata)

# # Step 7: Embed and normalize
# if documents:
#     logger.info(f"Embedding {len(documents)} chunks.")
#     raw_embeddings = embedding_fn(documents)
#     norm_embeddings = [normalize(e).tolist() for e in raw_embeddings]

#     # Step 8: Add to ChromaDB
#     collection.add(
#         documents=documents,
#         metadatas=metadatas,
#         ids=ids,
#         embeddings=norm_embeddings
#     )
#     logger.info(f"Stored {len(ids)} normalized embeddings in ChromaDB.")
# else:
#     logger.warning("No valid documents to embed.")

# # Step 9: Persist to disk
# client.persist()
# logger.info(f"Persisted {len(ids)} documents to ChromaDB at {CHROMA_DB_DIR}.")




import os
import json
import shutil
import numpy as np
import logging
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants
DATA_PATH = "extraction-work/structured_data3.json"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "saas_optimization"

# ─── Helper ─────────────────────────────────────────────────────────────────────
def normalize(vec):
    """Return unit-length vector (for cosine similarity)."""
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

# Step 1: Delete previous DB if needed
def delete_chroma_db(path):
    if os.path.exists(path):
        logger.info(f"Deleting old ChromaDB at {path}")
        shutil.rmtree(path)
    else:
        logger.info("No existing ChromaDB directory found.")

delete_chroma_db(CHROMA_DB_DIR)

# Step 2: Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path="./chroma_db")

# Step 3: Configure OpenAI embeddings
embedding_fn = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# Step 4: Create or reset collection
try:
    client.delete_collection(name=COLLECTION_NAME)
    logger.info(f"Existing '{COLLECTION_NAME}' collection deleted.")
except Exception:
    logger.info(f"No existing '{COLLECTION_NAME}' collection to delete.")
collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

# Step 5: Load and prepare JSON data
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} items from {DATA_PATH}.")
except Exception as e:
    logger.error(f"Failed to load JSON data: {e}")
    raise

# Step 6: Chunk and prepare data
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with specified size and overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

ids, documents, metadatas, raw_embeddings = [], [], [], []
seen_ids = set()  # Track used IDs to avoid duplicates

for item_idx, item in enumerate(data):
    required_keys = ["title", "text", "content_description", "subcategory", "content_type", "url", "category"]
    if not all(k in item for k in required_keys):
        logger.warning(f"Skipping item missing required keys: {item.get('url', 'unknown')}")
        continue

    text = f"{item['title']} {item['text']} {item['content_description']}".strip()
    if not text:
        logger.warning(f"Skipping empty text for item: {item.get('url', 'unknown')}")
        continue

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        # Generate unique ID using item index and chunk index
        base_id = f"{item['url']}_{i}"
        unique_id = base_id
        suffix = 0
        while unique_id in seen_ids:
            suffix += 1
            unique_id = f"{base_id}_{suffix}"  # Append suffix to ensure uniqueness
        seen_ids.add(unique_id)

        metadata = {
            "title": item["title"],
            "url": item["url"],
            "category": item["category"],
            "subcategory": item["subcategory"],
            "content_type": item["content_type"],
            "chunk_index": i
        }
        ids.append(unique_id)
        documents.append(chunk)
        metadatas.append(metadata)

# Step 7: Embed and normalize
if documents:
    logger.info(f"Embedding {len(documents)} chunks.")
    raw_embeddings = embedding_fn(documents)
    norm_embeddings = [normalize(e).tolist() for e in raw_embeddings]

    # Step 8: Add to ChromaDB
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=norm_embeddings
    )
    logger.info(f"Stored {len(ids)} normalized embeddings in ChromaDB.")
else:
    logger.warning("No valid documents to embed.")
