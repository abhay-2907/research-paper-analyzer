import os
import json
from tqdm import tqdm

input_folder = "data/processed/extracted_text"
output_folder = "data/processed/chunks"

os.makedirs(output_folder, exist_ok=True)

def chunk_text(text, chunk_size=500, overlap=100):
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

for file_name in tqdm(os.listdir(input_folder)):
    if not file_name.endswith(".json"):
        continue

    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    with open(input_path, "r", encoding="utf-8") as f:
        paper_data = json.load(f)

    paper_id = paper_data["paper_id"]
    title = paper_data["title"]
    pages = paper_data["pages"]

    all_chunks = []

    for page in pages:
        page_num = page["page_num"]
        text = page["text"].strip()

        if not text:
            continue

        page_chunks = chunk_text(text, chunk_size=200, overlap=50)

        for idx, chunk in enumerate(page_chunks, start=1):
            all_chunks.append({
                "paper_id": paper_id,
                "title": title,
                "page_num": page_num,
                "chunk_id": f"{paper_id}_page{page_num}_chunk{idx}",
                "text": chunk
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)