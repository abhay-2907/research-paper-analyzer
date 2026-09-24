import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

client = Groq(api_key="Your_api_key")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.load("data/vector_store/cleaned_embeddings.npy")

with open("data/vector_store/cleaned_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def cosine_similarity(query_embedding, embeddings):
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.dot(embeddings_norm, query_norm)
    return scores

paper_id = input("Enter paper ID: ").strip()
query = input("Enter your question: ").strip()

# For high-level questions, search only early pages
filtered_embeddings = []
filtered_metadata = []

for i, chunk in enumerate(metadata):
    if chunk["paper_id"] == paper_id and chunk["page_num"] <= 2:
        filtered_embeddings.append(embeddings[i])
        filtered_metadata.append(chunk)

if len(filtered_metadata) == 0:
    print("No chunks found for this paper.")
    exit()

filtered_embeddings = np.array(filtered_embeddings)

query_embedding = embedding_model.encode(query)
scores = cosine_similarity(query_embedding, filtered_embeddings)

top_k = min(3, len(filtered_metadata))
top_indices = np.argsort(scores)[-top_k:][::-1]

top_chunks = []
source_pages = []

for idx in top_indices:
    chunk = filtered_metadata[idx]
    top_chunks.append(
        f"[Page {chunk['page_num']}] {chunk['text']}"
    )
    source_pages.append(chunk["page_num"])

context = "\n\n".join(top_chunks)

prompt = f"""
You are a research paper assistant.

Your task is to answer the user's question using ONLY the retrieved paper context below.

IMPORTANT RULES:
- Focus on high-level meaning from the abstract/introduction.
- Ignore equations, formulas, symbols, and technical notation unless the question explicitly asks for them.
- Summarize in simple clear language.
- If the context does not clearly answer the question, say:
  "I could not find a confident answer in the retrieved paper sections."
- Do not copy random fragments.
- Do not hallucinate.

Retrieved Context:
{context}

User Question:
{query}

Return:
1. A short direct answer in 2-4 lines.
2. Then one bullet called "Evidence" with a short supporting statement.
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You answer research paper questions only from provided context."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.1
)

answer = response.choices[0].message.content

print("\n===== ANSWER =====\n")
print(answer)

print("\n===== SOURCES =====")
print("Pages used:", sorted(set(source_pages)))

print("\n===== RETRIEVED CHUNKS =====\n")
for i, chunk in enumerate(top_chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk[:700])
    print("-" * 80)
