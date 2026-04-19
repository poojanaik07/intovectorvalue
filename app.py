import streamlit as st
import pandas as pd
import time
from model import get_embeddings
from utils import find_duplicates_faiss

st.set_page_config(page_title="Duplicate Detector", layout="wide")

st.title("🌍 Multilingual Duplicate Detector")
st.caption("Detect duplicate records across languages using semantic similarity")

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    start_time = time.time()

    # 🔥 Progress Bar
    progress = st.progress(0, text="🚀 Starting...")

    df = pd.read_csv(uploaded_file)
    progress.progress(10, text="📂 File loaded")

    st.subheader("📊 Data Preview")
    st.dataframe(df, use_container_width=True)

    # Column handling
    if 'text' in df.columns:
        texts = df['text'].tolist()
        text_col = 'text'
    else:
        text_col = df.columns[1]
        texts = df[text_col].tolist()

    # Normalize
    texts = [str(t).lower().strip() for t in texts]

    # 🔄 Embedding
    embed_start = time.time()
    with st.spinner("🔄 Generating embeddings..."):
        embeddings = get_embeddings(texts)
    embed_done = time.time()
    progress.progress(60, text="🧠 Embeddings ready")

    # 🔍 Duplicate detection
    faiss_start = time.time()
    with st.spinner("🔍 Finding duplicates..."):
        labels = find_duplicates_faiss(embeddings)
    faiss_done = time.time()
    progress.progress(90, text="🔍 Groups created")

    df['group'] = labels
    df['group_count'] = df.groupby('group')['group'].transform('count')

    # 📊 Metrics
    total_time = faiss_done - start_time
    speed = len(df) / total_time

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Records", len(df))
    col2.metric("⚡ Speed", f"{speed:.0f} rec/sec")
    col3.metric("🧠 Groups", df['group'].nunique())

    st.divider()

    # 🔍 Only duplicates (sorted)
    duplicates = df[df['group_count'] > 1].copy()
    duplicates = duplicates.sort_values(by='group')

    st.subheader("🔍 Duplicate Records Only")
    st.dataframe(duplicates, use_container_width=True)

    st.divider()

    # 🔎 GROUP EXPLORER (FIXED)
    st.subheader("🔎 Explore Duplicate Groups")

    if len(duplicates) > 0:

        # ✅ SORTED + ALL GROUPS
        group_ids = sorted(duplicates['group'].unique())

        selected_group = st.selectbox("Select a group", group_ids)

        group_data = duplicates[duplicates['group'] == selected_group]

        st.write(f"### 📂 Group {selected_group} ({len(group_data)} items)")
        st.dataframe(group_data, use_container_width=True)

        # 🧠 CLEAN COMPARISON VIEW
        st.write("### 🧠 Comparison View")

        texts_in_group = group_data[text_col].tolist()

        seen = set()
        for i in range(len(texts_in_group)):
            for j in range(i + 1, len(texts_in_group)):
                pair = tuple(sorted((texts_in_group[i], texts_in_group[j])))
                if pair not in seen:
                    st.write(f"• {pair[0]}  ↔  {pair[1]}")
                    seen.add(pair)

    else:
        st.info("No duplicate groups found.")

    progress.progress(100, text="✅ Done!")

    st.divider()

    # ⏱️ Performance
    st.subheader("⏱️ Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Embedding Time", f"{embed_done - embed_start:.2f}s")
    col2.metric("🔍 Detection Time", f"{faiss_done - faiss_start:.2f}s")
    col3.metric("⏱️ Total Time", f"{total_time:.2f}s")

    # 📥 Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="duplicate_results.csv",
        mime="text/csv"
    )

    st.success("🚀 Duplicate detection completed successfully!")
    
