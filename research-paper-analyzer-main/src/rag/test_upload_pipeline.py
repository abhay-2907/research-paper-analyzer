import sys
import os
sys.path.append(os.path.abspath("."))

from src.rag.upload_pipeline import process_uploaded_pdf

pdf_path = "data/raw/pdfs/1803.01837v1.pdf"   # use one PDF you already have

chunks, embeddings = process_uploaded_pdf(pdf_path)

print("Total chunks:", len(chunks))
print("Embeddings shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])