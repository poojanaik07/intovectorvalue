import streamlit as st
import pandas as pd
import time
from model import get_embeddings
from utils import find_duplicates_faiss
from translator import translate_batch

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
    st.dataframe(df.head(500), use_container_width=True)

    # Column handling
    if 'text' in df.columns:
        text_col = 'text'
    else:
        text_col = df.columns[1]

    texts = df[text_col].astype(str).str.lower().str.strip().tolist()

    # 🔄 Embeddings
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

    # 🔍 Duplicate records only
    duplicates = df[df['group_count'] > 1].copy().sort_values(by='group')

    st.subheader("🔍 Duplicate Records Only")
    st.dataframe(duplicates.head(1000), use_container_width=True)

    st.divider()

    # 🔎 GROUP EXPLORER
    st.subheader("🔎 Explore Duplicate Groups")

    if len(duplicates) > 0:
        group_ids = sorted(duplicates['group'].unique())

        selected_group = st.selectbox("Select a group", group_ids)

        group_data = duplicates[duplicates['group'] == selected_group]

        st.write(f"### 📂 Group {selected_group} ({len(group_data)} items)")
        st.dataframe(group_data, use_container_width=True)

        # 🧠 Comparison View (limited for performance)
        st.write("### 🧠 Comparison View")

        texts_in_group = group_data[text_col].tolist()

        for i in range(min(3, len(texts_in_group))):
            for j in range(i + 1, min(3, len(texts_in_group))):
                st.write(f"• {texts_in_group[i]}  ↔  {texts_in_group[j]}")

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

    # 📥 Download original results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="duplicate_results.csv",
        mime="text/csv"
    )

    st.success("🚀 Duplicate detection completed successfully!")

    # 🌐 TRANSLATION FEATURE
    st.divider()
    st.subheader("🌐 Translate Dataset")

    lang_map = {
        "English": "en",
        "Hindi": "hi",
        "Japanese": "ja",
        "French": "fr",
        "German": "de",
        "Spanish": "es"
    }

    selected_lang = st.selectbox(
        "Select target language",
        list(lang_map.keys())
    )

    max_rows = st.slider(
        "Limit rows for translation (demo safe)",
        min_value=1,
        max_value=len(df),
        value=min(300, len(df))
    )

    if st.button("🌐 Translate Dataset"):

        st.warning("⚠️ Translating... Large datasets may take time")

        sample_df = df.head(max_rows).copy()

        translated_texts = translate_batch(
            sample_df[text_col].astype(str).tolist(),
            lang_map[selected_lang]
        )

        sample_df[f"translated_{selected_lang.lower()}"] = translated_texts

        st.success("✅ Translation completed!")

        st.dataframe(sample_df, use_container_width=True)

        csv_trans = sample_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=f"📥 Download {selected_lang} Version",
            data=csv_trans,
            file_name=f"translated_{selected_lang.lower()}.csv",
            mime="text/csv"
        )

    # 🧹 CLEAN DATASET BUTTON (NEW FEATURE)
    st.divider()
    st.subheader("🧹 Clean Dataset")

    if st.button("🧹 Generate Production-Ready Dataset"):

        clean_df = df.drop_duplicates(subset=['group']).copy()
        clean_df = clean_df.drop(columns=['group', 'group_count'])

        st.success("✅ Clean dataset generated!")

        st.dataframe(clean_df.head(1000), use_container_width=True)

        csv_clean = clean_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Clean Dataset",
            data=csv_clean,
            file_name="clean_dataset.csv",
            mime="text/csv"
        )
