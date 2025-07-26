from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_chain
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "saas-optimization-index"

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Models and vectorstore
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = LangchainPinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

# Custom prompt
prompt_template = """
You are a helpful SaaS assistant. Use only the following context to answer the user's question. If no relevant context exists, say so.

Context:
{context}

Question: {question}
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Chain using 'stuff' method
qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

@app.post("/ask")
async def ask_query(request: QueryRequest):
    try:
        # Step 1: Get top k results with score
        results_with_score = vectorstore.similarity_search_with_score(request.query, k=10)

        # Step 2: Filter results with good cosine similarity (threshold = 0.6 → distance <= 0.4)
        good_matches = [(doc, score) for doc, score in results_with_score if score <= 0.4]

        if not good_matches:
            # Fallback if no good results
            fallback_prompt = f"""The user asked: "{request.query}"

We couldn't find relevant matches. Please provide a helpful answer based on your SaaS knowledge.
"""
            fallback_answer = llm.predict(fallback_prompt)
            return {
                "answer": fallback_answer,
                "sources": []
            }

        # Step 3: Build RAG response using filtered context
        filtered_docs = [doc for doc, _ in good_matches]
        answer = qa_chain.run(input_documents=filtered_docs, question=request.query)

        # Step 4: Cluster results by subcategory
        clustered_sources = {}
        for doc, _ in good_matches:
            meta = doc.metadata
            subcat = meta.get("subcategory", "Uncategorized")
            if subcat not in clustered_sources:
                clustered_sources[subcat] = []
            clustered_sources[subcat].append({
                "title": meta.get("title"),
                "url": meta.get("url"),
                "content_type": meta.get("content_type")
            })

        return {
            "answer": answer,
            "clusters": clustered_sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
