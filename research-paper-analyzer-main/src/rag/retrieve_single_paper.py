import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# load CLEANED embeddings and metadata
embeddings = np.load("data/vector_store/cleaned_embeddings.npy")

with open("data/vector_store/cleaned_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(query_embedding, embeddings):
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.dot(embeddings_norm, query_norm)
    return scores

paper_id = input("Enter paper ID: ").strip()
query = input("Enter your question: ").strip()

filtered_embeddings = []
filtered_metadata = []

for i, chunk in enumerate(metadata):
    if chunk["paper_id"] == paper_id and chunk["page_num"] <= 3:
        filtered_embeddings.append(embeddings[i])
        filtered_metadata.append(chunk)

if len(filtered_metadata) == 0:
    print("No chunks found for this paper ID in first 3 pages.")
    exit()

filtered_embeddings = np.array(filtered_embeddings)

query_embedding = model.encode(query)
scores = cosine_similarity(query_embedding, filtered_embeddings)

top_k = min(5, len(filtered_metadata))
top_indices = np.argsort(scores)[-top_k:][::-1]

print("\nTop matching chunks from selected paper (pages 1-3 only):\n")

for rank, idx in enumerate(top_indices, start=1):
    chunk = filtered_metadata[idx]
    print(f"Rank {rank}")
    print(f"Paper ID   : {chunk['paper_id']}")
    print(f"Title      : {chunk['title']}")
    print(f"Page Number: {chunk['page_num']}")
    print(f"Chunk ID   : {chunk['chunk_id']}")
    print(f"Score      : {scores[idx]:.4f}")
    print(f"Text       : {chunk['text'][:500]}")
    print("-" * 80)