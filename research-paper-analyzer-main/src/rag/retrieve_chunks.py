import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# paths
embedding_path = "data/vector_store/embeddings.npy"
metadata_path = "data/vector_store/metadata.pkl"

# load saved data
embeddings = np.load(embedding_path)

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(query_embedding, embeddings):
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.dot(embeddings_norm, query_norm)
    return scores

# user query
query = input("Enter your question: ")

# convert query to embedding
query_embedding = model.encode(query)

# compute similarity scores
scores = cosine_similarity(query_embedding, embeddings)

# get top 5 results
top_k = 5
top_indices = np.argsort(scores)[-top_k:][::-1]

print("\nTop matching chunks:\n")

for rank, idx in enumerate(top_indices, start=1):
    chunk = metadata[idx]
    print(f"Rank {rank}")
    print(f"Paper ID   : {chunk['paper_id']}")
    print(f"Title      : {chunk['title']}")
    print(f"Page Number: {chunk['page_num']}")
    print(f"Chunk ID   : {chunk['chunk_id']}")
    print(f"Score      : {scores[idx]:.4f}")
    print(f"Text       : {chunk['text'][:500]}")
    print("-" * 80)