import streamlit as st
from generate import answer
from usage import get_totals
from documents import process_document, delete_document, list_documents
import time

st.set_page_config(page_title="Search", page_icon="🔎")

st.title("wiki qna")
st.caption("Ask questions over Wikipedia with cited sources")


question = st.text_input("Ask a question:")
if st.button("Ask") and question:
    start = time.time()
    text, points = answer(question)
    elapsed = time.time() - start
    st.write(text)
    st.caption(f"Answered in {elapsed:.1f}s")
    st.subheader("Sources")
    for i, point in enumerate(points, start=1): 
         with st.expander(f"[{i}] {point.payload.get('title', point.payload.get('name', ''))}"):
            st.write(point.payload['text'])

totals = get_totals()
st.sidebar.metric("Total tokens", totals["input_tokens"] + totals["output_tokens"])
st.sidebar.metric("Queries", totals["query_n"])

st.sidebar.divider()
st.sidebar.subheader("Document Management")

uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if st.sidebar.button("Upload") and uploaded_file:
    try:
        doc_id = process_document(uploaded_file, uploaded_file.name)
        st.sidebar.success(f"Uploaded {uploaded_file.name}")
    except ValueError as e:
        st.sidebar.error(str(e))

st.sidebar.divider()
st.sidebar.subheader("Uploaded Documents")
docs = list_documents()
if not docs:
    st.sidebar.write("No uploaded documents.")
else:
    for doc in docs:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"**{doc['name']}**\n{doc['upload_date'][:10]}")
        if col2.button("Delete", key=doc["doc_id"]):
            delete_document(doc["doc_id"])
            st.rerun()