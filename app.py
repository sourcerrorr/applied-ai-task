import streamlit as st
from generate import answer
from usage import get_totals
from documents import process_document, delete_document, list_documents
import time

st.set_page_config(page_title="Search", page_icon="🔎")

st.title("Investigation Engine")

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

with st.form("ask_form"):
    question = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    with st.spinner("Searching..."):
        before = get_totals()
        start = time.time()
        text, points = answer(question)
        elapsed = time.time() - start
        after = get_totals()
        tokens = (after["input_tokens"] + after["output_tokens"]) - (before["input_tokens"] + before["output_tokens"])
    st.session_state.last_answer = {"text": text, "points": points, "elapsed": elapsed, "tokens": tokens}

if st.session_state.last_answer:
    result = st.session_state.last_answer
    st.write(result["text"])
    st.caption(f"Answered in {result['elapsed']:.1f}s · {result['tokens']} tokens")
    st.subheader("Sources")
    for i, point in enumerate(result["points"], start=1):
        with st.expander(f"[{i}] {point.payload.get('title', point.payload.get('name', ''))}"):
            st.write(point.payload['text'])

totals = get_totals()
st.sidebar.metric("Total tokens", totals["input_tokens"] + totals["output_tokens"])
st.sidebar.metric("Queries", totals["query_n"])

st.sidebar.divider()
st.sidebar.subheader("Document Management")

uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if st.sidebar.button("Upload") and uploaded_file:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.sidebar.error("File exceeds 10 MB limit.")
    else:
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