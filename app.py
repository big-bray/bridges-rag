"""Streamlit search UI for the Bridges Mathematical Art Archive."""

from __future__ import annotations

import streamlit as st
from qdrant_client import QdrantClient

from bridges_rag.embed.embedder import DEFAULT_MODEL_NAME, Embedder
from bridges_rag.index.qdrant import DEFAULT_COLLECTION, DEFAULT_URL, get_client
from bridges_rag.search.citation import format_citation
from bridges_rag.search.search import search


@st.cache_resource
def _embedder() -> Embedder:
    return Embedder(DEFAULT_MODEL_NAME)


@st.cache_resource
def _client() -> QdrantClient:
    return get_client(DEFAULT_URL)


st.set_page_config(page_title="BridgesRAG", page_icon="🔎", layout="centered")
st.title("BridgesRAG")
st.caption("Semantic search over the Bridges Mathematical Art Archive")

query = st.text_input("Search the 2025 proceedings", placeholder="e.g. hyperbolic crochet patterns")

with st.sidebar:
    st.header("Filters")
    author = st.text_input("Author (exact match)")
    year = st.number_input("Year", min_value=0, max_value=2100, value=0, step=1)
    top_k = st.slider("Number of results", min_value=1, max_value=25, value=10)

if query:
    with st.spinner("Searching..."):
        results = search(
            _client(),
            _embedder(),
            query,
            collection=DEFAULT_COLLECTION,
            top_k=top_k,
            author=author or None,
            year=int(year) or None,
        )

    if not results:
        st.info("No results. Try loosening the filters.")

    for result in results:
        chunk = result.chunk
        with st.container(border=True):
            st.markdown(f"**{format_citation(chunk)}**  \nscore: {result.score:.3f}")
            st.write(chunk.text)
