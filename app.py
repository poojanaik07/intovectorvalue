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
    st.dataframe(df, use_container_width=True)

    # Safe column handling
    if 'text' in df.columns:
        texts = df['text'].tolist()
    else:
        texts = df[df.columns[1]].tolist()

    # Normalize
    texts = [str(t).lower().strip() for t in texts]

    # Embeddings
    embed_start = time.time()
    with st.spinner("🔄 Generating embeddings..."):
        embeddings = get_embeddings(texts)
    embed_done = time.time()

    # FAISS
    faiss_start = time.time()
    with st.spinner("🔍 Finding duplicates..."):
        labels = find_duplicates_faiss(embeddings)
    faiss_done = time.time()

    df['group'] = labels

    # Group count
    df['group_count'] = df.groupby('group')['group'].transform('count')

    # ✅ FILTER DUPLICATES ONLY (FAST + CLEAN)
    duplicates = df[df['group_count'] > 1]

    st.write("### 🔍 Duplicate Groups (Filtered)")
    st.dataframe(duplicates.sort_values('group'), use_container_width=True)

    # ✅ GROUP-WISE VIEW (VERY IMPRESSIVE)
    st.write("### 📂 View by Group")

    unique_groups = duplicates['group'].unique()[:10]  # limit for performance

    for g in unique_groups:
        with st.expander(f"Group {g}"):
            st.dataframe(duplicates[duplicates['group'] == g])

    # 📊 Metrics
    total_time = faiss_done - start_time
    speed = len(df) / total_time

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Records", len(df))
    col2.metric("⚡ Speed", f"{speed:.0f} rec/sec")
    col3.metric("🧠 Groups", df['group'].nunique())

    # ⏱️ Timing
    st.write("### ⏱️ Performance")
    st.write(f"⚡ Embedding Time: {embed_done - embed_start:.2f}s")
    st.write(f"🔍 Detection Time: {faiss_done - faiss_start:.2f}s")
    st.write(f"⏱️ Total Time: {total_time:.2f}s")
