import sys
import os
sys.path.append(os.path.abspath("."))

from src.rag.upload_pipeline import process_uploaded_pdf, retrieve_from_uploaded_pdf

pdf_path = "data/raw/pdfs/1803.01837v1.pdf"

chunks, embeddings = process_uploaded_pdf(pdf_path)

query = "What problem does the paper address?"

top_chunks, source_pages = retrieve_from_uploaded_pdf(query, chunks, embeddings)

print("Source Pages:", source_pages)

print("\nTop Retrieved Chunks:\n")
for i, chunk in enumerate(top_chunks, start=1):
    print(f"Chunk {i} | Page {chunk['page_num']}")
    print(chunk["text"][:700])
    print("-" * 80)