import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/search"

st.set_page_config(page_title="Document Search Engine", layout="centered")

st.title("Hybrid Document Search Engine")
st.write("BM25 + Transformer Semantic Reranking")

query = st.text_input("Enter search query")

top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    if not query.strip():
        st.warning("Enter a query")
    else:
        with st.spinner("Searching..."):
            response = requests.get(
                API_URL,
                params={"query": query, "top_k": top_k},
                timeout=30
            )

        if response.status_code == 200:
            results = response.json()

            if not results:
                st.info("No results found")
            else:
                for i, r in enumerate(results, 1):
                    st.subheader(f"{i}. {r['title']}")
                    st.write(f"BM25 score: {r['bm25_score']:.3f}")
                    st.write(f"Semantic score: {r['semantic_score']:.3f}")
                    st.divider()
        else:
            st.error("API error")
