````
You are an expert Python developer.
Generate a robust Python script that:

1. Crawls public pages under https://thegood.com/, including the “Insights” and case studies sections.
2. Follows links within that domain (limit depth = 2 to stay focused), skips repeats.
3. Fetches each page’s HTML using `requests` (with a browser User‑Agent header).
4. Parses the page with BeautifulSoup to extract:
   - Page URL, title (`<title>`), headers (h1–h3), and main textual content (e.g. article body or insight sections).
5. Cleans the text: remove scripts, styles, nav/footers, collapse whitespace.
6. Splits content into semantically coherent chunks (~500 tokens/characters) using either:
   - RecursiveCharacterTextSplitter from LangChain, or
   - Custom paragraph/sentence chunker.
7. Converts chunks into a JSONL output; each entry:
   ```json
   {
     "id": "<slug>_<chunk_index>",
     "url": "<page URL>",
     "title": "<page title>",
     "chunk": "<chunk text>"
   }
````

8. (Optional) If OpenAI API key is provided as env var, also generate embeddings per chunk using `openai.Embedding.create(model="text-embedding-ada-002", input=chunk)`, and include `"embedding": [ ... ]` in the JSONL.
9. Ensure the output is ready for ingestion into a vector DB or semantic-search index like Pinecone, Weaviate, or RAG pipelines.

Include proper error handling, rate limiting (e.g., `time.sleep(1)` between requests), and logging progress.

```

---
```
