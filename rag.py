import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. Load a small sample from extracted Wikipedia articles
file_path = "wiki_articles.json"
documents = []

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for article in data[:20]:
    text = article.get("text", "")
    if text:
        documents.append(text)

print("Loaded documents:", len(documents))

# 2. Chunk the text
chunks = []
chunk_size = 500

for doc in documents:
    for i in range(0, len(doc), chunk_size):
        chunk = doc[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

chunks = chunks[:100]
print("Testing with chunks:", len(chunks))

# 3. Create embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = embedding_model.encode(
    chunks,
    batch_size=8,
    show_progress_bar=True
).tolist()

print("Embeddings created:", len(embeddings))

# 4. Store in Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_store")

collection_name = "rag_docs_test_small"

try:
    chroma_client.delete_collection(collection_name)
except Exception:
    pass

collection = chroma_client.get_or_create_collection(name=collection_name)

ids = [str(i) for i in range(len(chunks))]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print("Stored embeddings in Chroma:", len(ids))

# 5. Ask a question
question = input("\nAsk a question: ")

query_embedding = embedding_model.encode([question]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

context = "\n".join(results["documents"][0])

print("\nRetrieved context:\n")
print(context)

# 6. Generate answer
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.responses.create(
    model="gpt-4.1-mini",
    input=f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""
)

print("\nAnswer:\n")
print(response.output_text)
