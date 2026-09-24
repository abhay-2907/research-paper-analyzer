import os
import json
import re
from tqdm import tqdm

input_folder = "data/processed/chunks"
output_folder = "data/processed/cleaned_chunks"

os.makedirs(output_folder, exist_ok=True)

def is_bad_chunk(text):
    text = text.strip()

    # too short
    if len(text.split()) < 40:
        return True

    lower_text = text.lower()

    # obvious noisy sections
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

    # too many digits/symbols, too little normal language
    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = len(text)

    if total_chars == 0:
        return True

    alpha_ratio = alpha_chars / total_chars

    if alpha_ratio < 0.5:
        return True

    # chunks that look like citation lists
    if re.search(r"\[\d+\]", text) and len(text.split()) < 80:
        return True

    return False

for file_name in tqdm(os.listdir(input_folder)):
    if not file_name.endswith(".json"):
        continue

    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    cleaned_chunks = []

    for chunk in chunks:
        if not is_bad_chunk(chunk["text"]):
            cleaned_chunks.append(chunk)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_chunks, f, ensure_ascii=False, indent=2)