from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "saas-optimization-index"

# Create or connect to Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load JSON data
with open("extraction-work/structured_data3.json", "r") as f:
    data = json.load(f)

# Text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Embed and store in Pinecone
for item in data:
    # Combine relevant text fields
    text = f"{item['title']} {item['text']} {item['content_description']}".strip()
    if not text:
        continue
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings for each chunk
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)
        
        # Create metadata
        metadata = {
            "subcategory": item["subcategory"],
            "content_type": item["content_type"],
            "title": item["title"],
            "url": item["url"],
            "category": item["category"],
            "chunk_id": f"{item['url']}_{i}"
        }
        
        # Upsert to Pinecone
        index.upsert(vectors=[(metadata["chunk_id"], embedding, metadata)])

print("Embeddings created and stored in Pinecone.")