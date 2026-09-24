import os
import json
import fitz
import pandas as pd
from tqdm import tqdm

# paths
metadata_path = "data/raw/metadata/arxiv_transformer_papers.csv"
pdf_folder = "data/raw/pdfs"
output_folder = "data/processed/extracted_text"

# create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# read metadata csv
df = pd.read_csv(metadata_path)

for _, row in tqdm(df.iterrows(), total=len(df)):
    paper_id = str(row["paper_id"]).replace("/", "_")
    title = row["title"]

    pdf_path = os.path.join(pdf_folder, f"{paper_id}.pdf")
    output_path = os.path.join(output_folder, f"{paper_id}.json")

    # skip if PDF does not exist
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        continue

    # skip if already extracted
    if os.path.exists(output_path):
        print(f"Already extracted: {paper_id}")
        continue

    try:
        doc = fitz.open(pdf_path)

        pages_data = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            pages_data.append({
                "page_num": page_num + 1,
                "text": text.strip()
            })

        paper_data = {
            "paper_id": paper_id,
            "title": title,
            "pages": pages_data
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(paper_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error processing {paper_id}: {e}")