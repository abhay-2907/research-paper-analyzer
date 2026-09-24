import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

chunk_folder = "data/processed/cleaned_chunks"
output_folder = "data/vector_store"

model = SentenceTransformer("all-MiniLM-L6-v2")

all_texts = []
all_metadata = []

for file_name in tqdm(os.listdir(chunk_folder)):
    if not file_name.endswith(".json"):
        continue

    file_path = os.path.join(chunk_folder, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        text = chunk["text"].strip()
        if text:
            all_texts.append(text)
            all_metadata.append({
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "page_num": chunk["page_num"],
                "chunk_id": chunk["chunk_id"],
                "text": text
            })

print("Total cleaned chunks:", len(all_texts))

embeddings = model.encode(all_texts, show_progress_bar=True)

np.save(os.path.join(output_folder, "cleaned_embeddings.npy"), embeddings)

with open(os.path.join(output_folder, "cleaned_metadata.pkl"), "wb") as f:
    pickle.dump(all_metadata, f)

print("Cleaned embeddings and metadata saved successfully.")