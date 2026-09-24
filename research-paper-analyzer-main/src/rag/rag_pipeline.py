import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from groq import Groq


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Load models and data only once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.load(os.path.join(PROJECT_ROOT, "data", "vector_store", "cleaned_embeddings.npy"))

with open(os.path.join(PROJECT_ROOT, "data", "vector_store", "cleaned_metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

load_dotenv()

def get_groq_client():
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")

    return Groq(api_key=api_key)

client = get_groq_client()


def cosine_similarity(query_embedding, embeddings):
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.dot(embeddings_norm, query_norm)
    return scores


def retrieve_chunks(paper_id, query, max_page=2, top_k=3):
    filtered_embeddings = []
    filtered_metadata = []

    for i, chunk in enumerate(metadata):
        if chunk["paper_id"] == paper_id and chunk["page_num"] <= max_page:
            filtered_embeddings.append(embeddings[i])
            filtered_metadata.append(chunk)

    if len(filtered_metadata) == 0:
        return [], []

    filtered_embeddings = np.array(filtered_embeddings)

    query_embedding = embedding_model.encode(query)
    scores = cosine_similarity(query_embedding, filtered_embeddings)

    top_k = min(top_k, len(filtered_metadata))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    top_chunks = []
    source_pages = []

    for idx in top_indices:
        chunk = filtered_metadata[idx]
        top_chunks.append(chunk)
        source_pages.append(chunk["page_num"])

    return top_chunks, sorted(set(source_pages))


def generate_answer(query, top_chunks):
    if not top_chunks:
        return "No relevant chunks found for this paper."

    context = "\n\n".join(
        [f"[Page {chunk['page_num']}] {chunk['text']}" for chunk in top_chunks]
    )

    prompt = f"""
You are a research paper assistant.

Your task is to answer the user's question using ONLY the retrieved paper context below.

IMPORTANT RULES:
- Focus on high-level meaning from the abstract/introduction.
- Ignore equations, formulas, symbols, and technical notation unless the question explicitly asks for them.
- Summarize in simple clear language.
- If the context does not clearly answer the question, say:
  "I could not find a confident answer in the retrieved paper sections."
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

    return response.choices[0].message.content

import pandas as pd

def get_available_papers():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "metadata", "arxiv_transformer_papers.csv"))
    papers = []

    for _, row in df.iterrows():
        papers.append({
            "paper_id": row["paper_id"],
            "title": row["title"]
        })

    return papers