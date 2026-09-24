import sys
import os
import tempfile

# Get the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

import streamlit as st
from src.rag.rag_pipeline import retrieve_chunks, generate_answer, get_available_papers
from src.rag.upload_pipeline import (
    process_uploaded_pdf,
    retrieve_from_uploaded_pdf,
    generate_answer_from_uploaded_pdf,
    extract_uploaded_pdf_metadata
)

st.set_page_config(page_title="Research Paper Analyzer", layout="wide")

st.title(" Research Paper Analyzer")
st.write("Analyze built-in research papers or upload your own PDF.")

# -------------------------
# MODE SELECTION
# -------------------------
mode = st.radio("Choose Mode", ["Built-in Papers", "Upload PDF"])

# -------------------------
# COMMON QUESTION UI
# -------------------------
st.subheader("Quick Actions")

col1, col2, col3, col4 = st.columns(4)

if col1.button("Problem"):
    st.session_state["query"] = "What problem does the paper address?"

if col2.button("Method"):
    st.session_state["query"] = "What method does the paper propose?"

if col3.button("Results"):
    st.session_state["query"] = "What are the main results of the paper?"

if col4.button("Summary"):
    st.session_state["query"] = "Give a short summary of this paper."

if "query" not in st.session_state:
    st.session_state["query"] = "What problem does the paper address?"

# -------------------------
# BUILT-IN PAPERS MODE
# -------------------------
if mode == "Built-in Papers":
    papers = get_available_papers()

    paper_options = {
        f"{paper['title']} ({paper['paper_id']})": paper["paper_id"]
        for paper in papers
    }

    selected_paper_label = st.selectbox("Select a Paper", list(paper_options.keys()))
    paper_id = paper_options[selected_paper_label]

    st.write(f"**Selected Paper ID:** {paper_id}")

    query = st.text_area("Ask your question", value=st.session_state["query"], key="built_in_query")
    st.session_state["query"] = query

    if st.button("Get Answer", key="built_in_answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant chunks..."):
                top_chunks, source_pages = retrieve_chunks(paper_id, query)

            if not top_chunks:
                st.error("No relevant chunks found for this paper ID.")
            else:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(query, top_chunks)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source Pages")
                pretty_pages = ", ".join([f"Page {p}" for p in source_pages])
                st.write(pretty_pages)

                st.subheader("Retrieved Chunks")
                for i, chunk in enumerate(top_chunks, start=1):
                    with st.expander(f"Chunk {i} | Page {chunk['page_num']}"):
                        st.write(chunk["text"])

# -------------------------
# UPLOAD PDF MODE
# -------------------------
else:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    query = st.text_area("Ask your question", value=st.session_state["query"], key="upload_query")
    st.session_state["query"] = query

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")

        # initialize session state
        if "uploaded_pdf_name" not in st.session_state:
            st.session_state["uploaded_pdf_name"] = None
        if "uploaded_chunks" not in st.session_state:
            st.session_state["uploaded_chunks"] = None
        if "uploaded_embeddings" not in st.session_state:
            st.session_state["uploaded_embeddings"] = None
        if "uploaded_metadata" not in st.session_state:
            st.session_state["uploaded_metadata"] = None

        # process only if this is a new uploaded file
        if st.session_state["uploaded_pdf_name"] != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_path = tmp_file.name

            try:
                with st.spinner("Processing uploaded PDF..."):
                    chunks, embeddings = process_uploaded_pdf(temp_pdf_path)
                    pdf_metadata = extract_uploaded_pdf_metadata(temp_pdf_path, chunks)

                st.session_state["uploaded_pdf_name"] = uploaded_file.name
                st.session_state["uploaded_chunks"] = chunks
                st.session_state["uploaded_embeddings"] = embeddings
                st.session_state["uploaded_metadata"] = pdf_metadata

            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        if st.session_state["uploaded_metadata"] is not None:
            meta = st.session_state["uploaded_metadata"]
            st.subheader("Uploaded PDF Info")
            st.write(f"**Detected Title:** {meta['title']}")
            st.write(f"**Total Pages:** {meta['total_pages']}")
            st.write(f"**Total Chunks Indexed:** {meta['total_chunks']}")

        if st.button("Analyze Uploaded PDF", key="upload_answer"):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                chunks = st.session_state["uploaded_chunks"]
                embeddings = st.session_state["uploaded_embeddings"]

                with st.spinner("Retrieving relevant chunks..."):
                    top_chunks, source_pages = retrieve_from_uploaded_pdf(query, chunks, embeddings)

                if not top_chunks:
                    st.error("No relevant chunks found in uploaded PDF.")
                else:
                    with st.spinner("Generating answer..."):
                        answer = generate_answer_from_uploaded_pdf(query, top_chunks)

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Source Pages")
                    pretty_pages = ", ".join([f"Page {p}" for p in source_pages])
                    st.write(pretty_pages)

                    st.subheader("Retrieved Chunks")
                    for i, chunk in enumerate(top_chunks, start=1):
                        with st.expander(f"Chunk {i} | Page {chunk['page_num']}"):
                            st.write(chunk["text"])
    else:
        st.info("Please upload a PDF file to analyze.")