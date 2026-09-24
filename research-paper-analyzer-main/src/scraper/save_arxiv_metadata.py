import requests
import feedparser
import pandas as pd
import time

BASE_URL = "http://export.arxiv.org/api/query"

def fetch_arxiv_papers(search_query="all:transformer", start=0, max_results=10):
    url = f"{BASE_URL}?search_query={search_query}&start={start}&max_results={max_results}"
    
    response = requests.get(url)
    feed = feedparser.parse(response.text)

    papers = []

    for entry in feed.entries:
        paper = {
            "paper_id": entry.id.split("/abs/")[-1],
            "title": entry.title.replace("\n", " ").strip(),
            "summary": entry.summary.replace("\n", " ").strip(),
            "published": entry.published,
            "authors": ", ".join(author.name for author in entry.authors),
            "pdf_url": ""
        }

        for link in entry.links:
            if link.type == "application/pdf":
                paper["pdf_url"] = link.href
                break

        papers.append(paper)

    return papers


all_papers = []

for start in range(0, 50, 10):   # gets 50 papers in batches of 10
    papers = fetch_arxiv_papers(search_query="all:transformer", start=start, max_results=10)
    all_papers.extend(papers)
    time.sleep(3)   # polite delay so we don't spam arXiv

df = pd.DataFrame(all_papers)

output_path = "data/raw/metadata/arxiv_transformer_papers.csv"
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} papers to {output_path}")