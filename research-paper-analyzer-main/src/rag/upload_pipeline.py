import fitz
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from groq import Groq

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        pages.append({
            "page_num": page_num + 1,
            "text": text
        })

    return pages


def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def is_bad_chunk(text):
    text = text.strip()

    if len(text.split()) < 40:
        return True

    lower_text = text.lower()

    bad_starts = [
        "figure",
        "fig.",
        "table",
        "references",
        "appendix",
        "publication date"
    ]

    for bad in bad_starts:
        if lower_text.startswith(bad):
            return True

    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = len(text)

    if total_chars == 0:
        return True

    alpha_ratio = alpha_chars / total_chars

    if alpha_ratio < 0.5:
        return True

    if re.search(r"\[\d+\]", text) and len(text.split()) < 80:
        return True

    return False


def create_chunks_from_pages(pages):
    all_chunks = []

    for page in pages:
        page_num = page["page_num"]
        text = page["text"]

        if not text.strip():
            continue

        page_chunks = chunk_text(text, chunk_size=200, overlap=50)

        for idx, chunk in enumerate(page_chunks, start=1):
            if not is_bad_chunk(chunk):
                all_chunks.append({
                    "page_num": page_num,
                    "chunk_id": f"page{page_num}_chunk{idx}",
                    "text": chunk
                })

    return all_chunks


def create_embeddings_for_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts)
    return np.array(embeddings)


def process_uploaded_pdf(pdf_path):
    pages = extract_text_from_pdf(pdf_path)
    chunks = create_chunks_from_pages(pages)
    embeddings = create_embeddings_for_chunks(chunks)

    return chunks, embeddings


def cosine_similarity(query_embedding, embeddings):
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.dot(embeddings_norm, query_norm)
    return scores


def retrieve_from_uploaded_pdf(query, chunks, embeddings, top_k=3):
    query_lower = query.lower()

    # route query to likely relevant page ranges
    if any(word in query_lower for word in ["problem", "summary", "address", "overview"]):
        filtered_indices = [i for i, chunk in enumerate(chunks) if chunk["page_num"] <= 2]

    elif any(word in query_lower for word in ["method", "approach", "architecture", "model", "propose"]):
        filtered_indices = [i for i, chunk in enumerate(chunks) if chunk["page_num"] <= 4]

    elif any(word in query_lower for word in ["result", "results", "experiment", "performance", "accuracy"]):
        filtered_indices = [i for i, chunk in enumerate(chunks) if chunk["page_num"] >= 4]

    elif any(word in query_lower for word in ["dataset", "data", "benchmark"]):
        filtered_indices = [i for i, chunk in enumerate(chunks) if chunk["page_num"] >= 3]

    else:
        filtered_indices = list(range(len(chunks)))

    if len(filtered_indices) == 0:
        return [], []

    filtered_chunks = [chunks[i] for i in filtered_indices]
    filtered_embeddings = np.array([embeddings[i] for i in filtered_indices])

    query_embedding = embedding_model.encode(query)
    scores = cosine_similarity(query_embedding, filtered_embeddings)

    top_k = min(top_k, len(filtered_chunks))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    top_chunks = []
    source_pages = []

    for idx in top_indices:
        chunk = filtered_chunks[idx]
        top_chunks.append(chunk)
        source_pages.append(chunk["page_num"])

    return top_chunks, sorted(set(source_pages))



load_dotenv()

def get_groq_client():
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")

    return Groq(api_key=api_key)

client = get_groq_client()

def generate_answer_from_uploaded_pdf(query, top_chunks):
    if not top_chunks:
        return "No relevant chunks found in the uploaded PDF."

    context = "\n\n".join(
        [f"[Page {chunk['page_num']}] {chunk['text']}" for chunk in top_chunks]
    )

    prompt = f"""
You are a research paper assistant.

Your task is to answer the user's question using ONLY the retrieved paper context below.

IMPORTANT RULES:
- Focus on high-level meaning from the abstract/introduction when relevant.
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


def extract_uploaded_pdf_metadata(pdf_path, chunks):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    title = "Unknown Title"

    if total_pages > 0:
        first_page_text = doc[0].get_text()
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]

        title_lines = []

        for line in lines[:15]:  # only inspect the first few lines
            lower_line = line.lower()

            # stop if we reach abstract or author/email-like text
            if "abstract" in lower_line:
                break
            if "@" in line:
                break

            # skip very short lines
            if len(line.split()) < 2:
                continue

            # skip affiliation-like lines
            if any(word in lower_line for word in ["university", "research", "google", "institute", "department"]):
                continue

            title_lines.append(line)

            # usually title is 1–2 lines
            if len(title_lines) >= 2:
                break

        if title_lines:
            title = " ".join(title_lines)

            # clean extra spacing
            title = " ".join(title.split())

            # if title is too long, trim
            if len(title) > 150:
                title = title[:150] + "..."

    return {
        "title": title,
        "total_pages": total_pages,
        "total_chunks": len(chunks)
    }