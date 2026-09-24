import sys
import os
sys.path.append(os.path.abspath("."))

from src.rag.upload_pipeline import (
    process_uploaded_pdf,
    retrieve_from_uploaded_pdf,
    generate_answer_from_uploaded_pdf
)

pdf_path = "data/raw/pdfs/1803.01837v1.pdf"
query = "What problem does the paper address?"

chunks, embeddings = process_uploaded_pdf(pdf_path)
top_chunks, source_pages = retrieve_from_uploaded_pdf(query, chunks, embeddings)
answer = generate_answer_from_uploaded_pdf(query, top_chunks)

print("ANSWER:\n")
print(answer)

print("\nSOURCES:")
print(source_pages)

print("\nTOP CHUNKS:")
for i, chunk in enumerate(top_chunks, start=1):
    print(f"\nChunk {i} | Page {chunk['page_num']}")
    print(chunk["text"][:700])