import os
import pandas as pd
import requests
import time
from tqdm import tqdm

metadata_path = "data/raw/metadata/arxiv_transformer_papers.csv"
pdf_folder = "data/raw/pdfs"

os.makedirs(pdf_folder, exist_ok=True)

df = pd.read_csv(metadata_path)

for _, row in tqdm(df.iterrows(), total=len(df)):
    paper_id = str(row["paper_id"]).replace("/", "_")
    pdf_url = row["pdf_url"]

    if not isinstance(pdf_url, str) or pdf_url.strip() == "":
        continue

    pdf_path = os.path.join(pdf_folder, f"{paper_id}.pdf")

    if os.path.exists(pdf_path):
        print(f"Already exists: {paper_id}")
        continue

    try:
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {paper_id}")
        else:
            print(f"Failed: {paper_id} | Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {paper_id}: {e}")

    time.sleep(2)   # polite delay