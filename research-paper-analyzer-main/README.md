# Research Paper Analyzer

Live Demo: [Try the app here](https://piyushsingghh-research-paper-analyzer-srcappinterface-2n1fyy.streamlit.app/))

An AI-powered research paper analysis tool, which allows users to upload any research paper in PDF format or choose from attached set of arXiv research papers, and ask natural language questions about them.

# What Problem Does This Solve?

Reading research papers can be boring, especially when you want to find answers to a specific questions in the paper.

**What can I do with the Research Paper Analyzer?**

1- Upload a research paper in PDF format.
2- Ask questions about the research paper in English.
3- Receive answers to those questions in English, along with page numbers from the research paper.

No need to scroll through a 15-page research paper to find one specific answer.

---

# Key Features

1- **Two modes:**
  - Pick from a curated collection of transformer and NLP research papers that have already been indexed from arXiv.
  - Upload your research paper in PDF format.

2- **Answer questions about the research paper:**
  - Answers are based on the content of the research paper.
  - Answers are supported by page numbers from the research paper.
  - Retrieved chunks of the research paper are displayed.

3- **Quick action buttons:**
  - Problem: What problem is the research paper solving?
  - Method: What method does the research paper propose?
  - Results: What are the results?
  - Summary: Give a short summary.

4- **Question-aware search:**
  - Problem/Summary: search abstract, introduction, etc.
  - Method: search early, mid, etc.
  - Results: search experiment, etc.
  - This increases the accuracy of the answer.

5- **Cache the results:**
  - Results are computed once when the research paper is uploaded.
  - If the user poses the same question again, the answer will appear quickly.

---

##  How It Works

This project uses a **RAG (Retrieval-Augmented Generation) pipeline**


### Step-by-step:

1. **Text Extraction:** PyMuPDF extracts text page-by-page from PDFs
2. **Chunking:** Long text is split into ~200-word chunks with 50-word overlap
3. **Cleaning:** Noisy chunks (figure captions, references, symbols) are filtered out
4. **Embeddings:** Each chunk is converted to a 384-dim vector using SentenceTransformers
5. **Retrieval:** User query is embedded and compared with chunk embeddings using cosine similarity
6. **Answer Generation:** Top relevant chunks are sent to an LLM (LLaMA 3.1 via Groq) with a carefully designed prompt
7. **Citation:** Source page numbers and evidence chunks are displayed alongside the answer

---

##  Tech Stack

 UI  - Streamlit 
 PDF Parsing  - PyMuPDF (fitz) 
 Embeddings  - SentenceTransformers (all-MiniLM-L6-v2) 
 Similarity Search  - NumPy cosine similarity 
 LLM  - LLaMA 3.1 8B via Groq API 
 Data Collection - arXiv API 
 Language - Python 

---

## Project Structure

research-paper-analyzer/
│
├── data/
│ ├── raw/
│ │ ├── metadata/ # arXiv paper metadata CSV
│ │ └── pdfs/ # downloaded paper PDFs
│ ├── processed/
│ │ ├── extracted_text/ # page-wise extracted text
│ │ ├── chunks/ # chunked text
│ │ └── cleaned_chunks/ # cleaned chunks
│ └── vector_store/ # embeddings and metadata
│
├── src/
│ ├── app/
│ │ └── interface.py # Streamlit UI
│ ├── parser/
│ │ ├── extract_all_pdfs.py
│ │ ├── chunk_papers.py
│ │ └── clean_chunks.py
│ ├── rag/
│ │ ├── rag_pipeline.py # built-in paper retrieval + answer
│ │ ├── upload_pipeline.py # uploaded PDF processing
│ │ └── create_embeddings.py
│ └── scraper/
│ ├── save_arxiv_metadata.py
│ └── download_pdfs.py
│
├── .gitignore
├── requirements.txt
└── README.md
