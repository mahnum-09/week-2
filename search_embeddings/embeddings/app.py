from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("App started...")

documents = [
    "Machine learning is amazing",
    "Deep learning is part of artificial intelligence",
    "Dogs are friendly animals",
    "Cats are beautiful pets",
    "I love studying data science",
    "Artificial intelligence will change the future"
]

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(documents)

print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype="float32"))

print("App ready.")

def search_query(query, k=3):
    print("Searching...")
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype="float32"), k)
    results = [documents[i] for i in indices[0]]
    return results