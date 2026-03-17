import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load extracted Wikipedia articles
file_path = "wiki_articles.json"
documents = []

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Small test sample for now
for article in data[:20]:
    text = article.get("text", "")
    if text and not text.startswith("#REDIRECT"):
        documents.append(text)

print("Loaded documents:", len(documents))

# Chunk documents
chunks = []
chunk_size = 500

for doc in documents:
    for i in range(0, len(doc), chunk_size):
        chunk = doc[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

chunks = chunks[:100]
print("Building chunks:", len(chunks))

# Create embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = embedding_model.encode(
    chunks,
    batch_size=8,
    show_progress_bar=True
).tolist()

print("Embeddings created:", len(embeddings))

# Store in persistent Chroma
db_path = "./chroma_store"
collection_name = "wiki_rag"

chroma_client = chromadb.PersistentClient(path=db_path)

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
print("Database build complete.")
