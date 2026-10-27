import streamlit as st
from generate import answer
from usage import get_totals

st.set_page_config(page_title="Search", page_icon="🔎")

st.title("wiki qna")
st.caption("Ask questions over Wikipedia with cited sources")

question = st.text_input("Ask a question:")
if st.button("Ask") and question:
    text, points = answer(question)
    st.write(text)
    st.subheader("Sources")
    for i, point in enumerate(points, start=1): 
         with st.expander(f"[{i}] {point.payload['title']}"):
            st.write(point.payload['text'])

totals = get_totals()
st.sidebar.metric("Total tokens", totals["input_tokens"] + totals["output_tokens"])
st.sidebar.metric("Queries", totals["query_n"])