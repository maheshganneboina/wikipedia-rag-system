import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

db_path = "./chroma_store"
collection_name = "wiki_rag"

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load existing Chroma DB
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name=collection_name)

print("RAG system ready. Type 'exit' to quit.\n")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

while True:
    question = input("Ask a question: ").strip()

    if question.lower() == "exit":
        break

    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    print("\nRetrieved context:\n")
    print(context)

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
    print("\n" + "-" * 60 + "\n")
