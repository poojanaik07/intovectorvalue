import streamlit as st
import pandas as pd
from model import get_embeddings
from utils import find_duplicates_faiss
import time

st.title("🌍 Multilingual Duplicate Detector")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    start_time = time.time()
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df)

    # Safe column handling
    if 'text' in df.columns:
        texts = df['text'].tolist()
    else:
        texts = df[df.columns[1]].tolist()

    # Normalize text
    texts = [str(t).lower().strip() for t in texts]

    # Embeddings
    embed_start = time.time()
    with st.spinner("🔄 Generating embeddings..."):
        embeddings = get_embeddings(texts)
    embed_done = time.time()
    faiss_start = time.time()
    # FAISS grouping
    with st.spinner("🔍 Finding duplicates..."):
        labels = find_duplicates_faiss(embeddings)
    faiss_done = time.time()
    df['group'] = labels

    st.write("### ✅ Duplicate Groups")
    st.dataframe(df.sort_values('group'))
     # 📊 SHOW TIMES
    st.write(f"⚡ Embedding Time: {embed_done - embed_start:.2f}s")
    st.write(f"🔍 Duplicate Detection Time: {faiss_done - faiss_start:.2f}s")
    st.write(f"⏱️ Total Time: {faiss_done - start_time:.2f}s")
