import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import get_embeddings
from utils import find_duplicates_faiss

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual Duplicate Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    padding-top: 1rem;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Nav buttons */
div.stButton > button {
    width: 100%; text-align: left;
    padding: 0.55rem 1rem; border-radius: 10px;
    border: none; background: transparent;
    color: #94a3b8 !important; font-size: 0.93rem; font-weight: 500;
    transition: all 0.2s; cursor: pointer;
}
div.stButton > button:hover { background: #334155 !important; color: #f1f5f9 !important; }

/* KPI metric cards */
[data-testid="stMetric"] {
    background: #ffffff; border-radius: 14px;
    padding: 1.1rem 1.3rem; border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"]  { font-size: 0.8rem !important; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
[data-testid="stMetricValue"]  { font-size: 1.9rem !important; font-weight: 700; color: #0f172a !important; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

/* Chart container */
.chart-card {
    background: #ffffff; border-radius: 16px; padding: 1.2rem;
    border: 1px solid #e8edf3; box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

hr { border-color: #e2e8f0; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────
@st.cache_data
def cached_embeddings(texts):
    return get_embeddings(texts)

@st.cache_data
def cached_groups(embeddings):
    return find_duplicates_faiss(embeddings)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for key, default in {
    "active_tab": "📂 Upload",
    "df": None, "clean_df": None, "duplicates": None,
    "text_col": None, "embed_time": 0.0,
    "detect_time": 0.0, "total_time": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
LAYOUT  = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=12, color="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", y=-0.15),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
TABS = [
    ("📂 Upload",      "Upload & Process"),
    ("📈 Dashboard",   "Analytics Dashboard"),
    ("📊 Data Preview","Raw Data Preview"),
    ("🔍 Duplicates",  "Duplicate Explorer"),
    ("🧹 Clean Dataset","Clean Dataset"),
    ("🌐 Translate",   "Translate Dataset"),
]

with st.sidebar:
    st.markdown("## 🌍 Duplicate Detector")
    st.markdown("---")
    for tab_key, tab_label in TABS:
        is_active = st.session_state.active_tab == tab_key
        prefix = "▶ " if is_active else "   "
        if st.button(f"{prefix}{tab_label}", key=f"nav_{tab_key}"):
            st.session_state.active_tab = tab_key
    st.markdown("---")
    if st.session_state.df is not None:
        df = st.session_state.df
        cdf = st.session_state.clean_df
        st.success("✅ File loaded")
        st.caption(f"📄 {len(df)} records")
        st.caption(f"🧹 {len(cdf)} clean records")
        st.caption(f"⏱ {st.session_state.total_time:.1f}s")
    else:
        st.info("Upload a CSV to begin")

active = st.session_state.active_tab

# ═══════════════════════════════════════════════════════════
# TAB — UPLOAD
# ═══════════════════════════════════════════════════════════
if active == "📂 Upload":
    st.title("📂 Upload & Process")
    st.caption("Upload a CSV file to detect multilingual duplicates using semantic similarity.")
    st.divider()

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_col = 'text' if 'text' in df.columns else df.columns[1]
        texts = df[text_col].astype(str).str.lower().str.strip().tolist()

        progress = st.progress(0, text="🚀 Starting pipeline...")
        start_time = time.time()

        embed_start = time.time()
        with st.spinner("🔄 Generating embeddings..."):
            embeddings = cached_embeddings(texts)
        embed_done = time.time()
        progress.progress(60, text="🧠 Embeddings ready")

        faiss_start = time.time()
        with st.spinner("🔍 Finding duplicates..."):
            labels = cached_groups(embeddings)
        faiss_done = time.time()
        progress.progress(90, text="🔍 Groups created")

        df['group'] = labels
        clean_df   = df.drop_duplicates(subset=['group']).copy().drop(columns=['group'])
        duplicates = df[df.duplicated(subset=['group'], keep=False)].copy().sort_values('group')
        total_time = faiss_done - start_time
        progress.progress(100, text="✅ Done!")

        st.session_state.update({
            "df": df, "clean_df": clean_df, "duplicates": duplicates,
            "text_col": text_col,
            "embed_time": embed_done - embed_start,
            "detect_time": faiss_done - faiss_start,
            "total_time": total_time,
        })

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📄 Total Records",  len(df))
        c2.metric("🧹 Clean Records",  len(clean_df))
        c3.metric("🔁 Duplicates",     len(df) - len(clean_df))
        c4.metric("⚡ Speed",          f"{len(df)/total_time:.0f} rec/s")

        st.success("✅ Done! Click **📈 Analytics Dashboard** in the sidebar to explore charts.")

        csv_raw = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Raw Results", csv_raw,
                           file_name="duplicate_results.csv", mime="text/csv")

    elif st.session_state.df is not None:
        st.info("✅ File already loaded. Use the sidebar to navigate.")
    else:
        st.markdown("""
        ### 👋 Welcome!
        Upload a CSV file to get started.

        **What this tool does:**
        - 🔍 Detects duplicate records across **multiple languages**
        - 📈 Auto-generates **Power BI-style analytics charts**
        - 🧹 Generates a **clean deduplicated** dataset
        - 🌐 **Translates** cleaned data to your preferred language
        """)

# ═══════════════════════════════════════════════════════════
# TAB — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════
elif active == "📈 Dashboard":
    st.title("📈 Analytics Dashboard")
    st.caption("Auto-generated visual insights from your uploaded dataset — Power BI style.")
    st.divider()

    if st.session_state.df is None:
        st.warning("⚠️ No file loaded yet. Go to **📂 Upload** first.")
    else:
        df         = st.session_state.df.copy()
        clean_df   = st.session_state.clean_df.copy()
        duplicates = st.session_state.duplicates.copy()
        text_col   = st.session_state.text_col

        total     = len(df)
        n_clean   = len(clean_df)
        n_dup     = total - n_clean
        dup_pct   = round(n_dup / total * 100, 1) if total else 0

        # ── KPI Row ──────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("📄 Total Records",   total)
        k2.metric("🧹 Clean Records",   n_clean, delta=f"-{n_dup} removed")
        k3.metric("🔁 Duplicates",      n_dup,   delta=f"{dup_pct}% of total", delta_color="inverse")
        k4.metric("⚡ Embed Time",      f"{st.session_state.embed_time:.1f}s")
        k5.metric("🔍 Detection Time",  f"{st.session_state.detect_time:.1f}s")

        st.divider()

        # ── ROW 1: Pie  +  Duplicate Group Size Bar ──────────
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown("#### 🥧 Dataset Composition")
            pie_fig = px.pie(
                values=[n_clean, n_dup],
                names=["Clean Records", "Duplicates"],
                color_discrete_sequence=["#3b82f6", "#f97316"],
                hole=0.55,
            )
            pie_fig.update_traces(textposition="outside", textinfo="percent+label",
                                  pull=[0.04, 0.04])
            pie_fig.update_layout(**LAYOUT, height=340,
                                  annotations=[dict(text=f"<b>{total}</b><br>total",
                                                    x=0.5, y=0.5, font_size=16,
                                                    showarrow=False)])
            st.plotly_chart(pie_fig, use_container_width=True)

        with r1c2:
            st.markdown("#### 📦 Duplicate Group Sizes")
            if 'group' in df.columns:
                grp_sizes = df.groupby('group').size().reset_index(name='count')
                vc        = grp_sizes['count'].value_counts().sort_index()
                grp_hist  = pd.DataFrame({
                    'Group Size':       vc.index.tolist(),
                    'Number of Groups': vc.values.tolist()
                })
                bar_grp = px.bar(grp_hist, x='Group Size', y='Number of Groups',
                                 color='Number of Groups',
                                 color_continuous_scale='Blues',
                                 text='Number of Groups')
                bar_grp.update_traces(textposition='outside')
                bar_grp.update_layout(**LAYOUT, height=340, showlegend=False,
                                      coloraxis_showscale=False)
                st.plotly_chart(bar_grp, use_container_width=True)
            else:
                st.info("Group info not available.")

        st.divider()

        # ── Auto-detect categorical & numeric columns ─────────
        # Exclude internal/id-like columns
        skip_cols = {'group', 'id', 'ID', 'index'}
        cat_cols  = [c for c in df.columns
                     if df[c].dtype == object and c not in skip_cols
                     and df[c].nunique() <= 40]
        num_cols  = [c for c in df.columns
                     if pd.api.types.is_numeric_dtype(df[c]) and c not in skip_cols
                     and c != 'group']

        # ── ROW 2: Dynamic single-chart explorer ──────────────
        if cat_cols:
            st.markdown("#### 📊 Top Value Distributions")

            # ── Filter controls in one compact row ──
            fc1, fc2, fc3 = st.columns([2, 1, 1])
            with fc1:
                sel_col = st.selectbox("📋 Select column", cat_cols, key="dist_col")
            with fc2:
                max_unique = min(df[sel_col].nunique(), 30)
                top_n = st.slider("🔢 Top N values", 3, max_unique,
                                  min(10, max_unique), key="dist_topn")
            with fc3:
                chart_type = st.radio("📐 Chart type",
                                      ["Horizontal Bar", "Vertical Bar", "Pie"],
                                      horizontal=True, key="dist_type")

            # ── Build data ──
            vc = df[sel_col].value_counts().head(top_n)
            dist_df = pd.DataFrame({"Value": vc.index.tolist(),
                                    "Count": vc.values.tolist()})

            # ── Render chosen chart ──
            if chart_type == "Horizontal Bar":
                dist_fig = px.bar(dist_df, x="Count", y="Value",
                                  orientation="h",
                                  color="Count",
                                  color_continuous_scale="Teal",
                                  text="Count",
                                  title=f"Top {top_n} values — {sel_col}")
                dist_fig.update_traces(textposition="outside")
                dist_fig.update_layout(**LAYOUT, height=420, showlegend=False,
                                       coloraxis_showscale=False,
                                       yaxis=dict(autorange="reversed"))

            elif chart_type == "Vertical Bar":
                dist_fig = px.bar(dist_df, x="Value", y="Count",
                                  color="Count",
                                  color_continuous_scale="Purples",
                                  text="Count",
                                  title=f"Top {top_n} values — {sel_col}")
                dist_fig.update_traces(textposition="outside")
                dist_fig.update_layout(**LAYOUT, height=420, showlegend=False,
                                       coloraxis_showscale=False,
                                       xaxis_tickangle=-35)

            else:  # Pie
                dist_fig = px.pie(dist_df, values="Count", names="Value",
                                  hole=0.4,
                                  color_discrete_sequence=px.colors.qualitative.Pastel,
                                  title=f"Top {top_n} values — {sel_col}")
                dist_fig.update_traces(textposition="outside",
                                       textinfo="percent+label")
                dist_fig.update_layout(**LAYOUT, height=420)

            st.plotly_chart(dist_fig, use_container_width=True)
            st.divider()

        # ── ROW 3: Histograms for numeric cols ────────────────
        if num_cols:
            st.markdown("#### 📉 Numeric Column Distributions")
            cols_per_row = 2
            for i in range(0, len(num_cols), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col in enumerate(num_cols[i:i+cols_per_row]):
                    with row_cols[j]:
                        fig = px.histogram(df, x=col, nbins=25,
                                           color_discrete_sequence=['#6366f1'],
                                           title=f"Distribution — {col}",
                                           marginal="box")
                        fig.update_layout(**LAYOUT, height=340)
                        st.plotly_chart(fig, use_container_width=True)
            st.divider()

        # ── ROW 4: Duplicate rate per category (if cat cols) ──
        if cat_cols and 'group' in df.columns:
            st.markdown("#### 🔁 Duplicate Rate by Category")
            dup_rate_col = cat_cols[0]          # use first categorical column
            tab_sel = st.selectbox("Select column for duplicate rate breakdown",
                                   cat_cols, key="dup_rate_sel")

            dup_flag = df.duplicated(subset=['group'], keep=False)
            df['_is_dup'] = dup_flag

            rate_df = (df.groupby(tab_sel)['_is_dup']
                       .agg(['sum','count'])
                       .reset_index())
            rate_df.columns = [tab_sel, 'duplicates', 'total']
            rate_df['dup_rate_%'] = (rate_df['duplicates'] / rate_df['total'] * 100).round(1)
            rate_df = rate_df.sort_values('dup_rate_%', ascending=False).head(20)

            rc1, rc2 = st.columns(2)
            with rc1:
                fig_rate = px.bar(rate_df, x=tab_sel, y='dup_rate_%',
                                  color='dup_rate_%',
                                  color_continuous_scale='OrRd',
                                  title=f"Duplicate % by {tab_sel}",
                                  text='dup_rate_%')
                fig_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_rate.update_layout(**LAYOUT, height=380, showlegend=False,
                                       coloraxis_showscale=False,
                                       xaxis_tickangle=-35)
                st.plotly_chart(fig_rate, use_container_width=True)

            with rc2:
                fig_tree = px.treemap(rate_df,
                                      path=[tab_sel],
                                      values='total',
                                      color='dup_rate_%',
                                      color_continuous_scale='RdBu_r',
                                      title=f"Volume vs Duplicate Rate — {tab_sel}")
                fig_tree.update_layout(**LAYOUT, height=380)
                st.plotly_chart(fig_tree, use_container_width=True)

            df.drop(columns=['_is_dup'], inplace=True)
            st.divider()

        # ── ROW 5: Scatter — if ≥2 numeric cols ──────────────
        if len(num_cols) >= 2:
            st.markdown("#### 🔵 Correlation Scatter")
            sc1, sc2, sc3 = st.columns(3)
            x_col   = sc1.selectbox("X axis",   num_cols, index=0, key="sc_x")
            y_col   = sc2.selectbox("Y axis",   num_cols, index=min(1, len(num_cols)-1), key="sc_y")
            col_col = sc3.selectbox("Color by", ["None"] + cat_cols, key="sc_c")

            scatter_kwargs = dict(x=x_col, y=y_col, opacity=0.7,
                                  color_discrete_sequence=PALETTE)
            if col_col != "None":
                scatter_kwargs["color"] = col_col

            # Add OLS trendline only if statsmodels is available
            try:
                import statsmodels  # noqa
                scatter_kwargs["trendline"] = "ols"
            except ModuleNotFoundError:
                pass

            fig_sc = px.scatter(df, **scatter_kwargs,
                                title=f"{y_col} vs {x_col}")
            fig_sc.update_layout(**LAYOUT, height=420)
            st.plotly_chart(fig_sc, use_container_width=True)
            st.divider()

        # ── ROW 6: Clean vs Original column comparison ────────
        if cat_cols:
            st.markdown("#### ✅ Before vs After Deduplication")
            comp_col = st.selectbox("Compare column", cat_cols, key="comp_col")

            vc_orig = df[comp_col].value_counts().head(15)
            original_counts = pd.DataFrame({comp_col: vc_orig.index.tolist(), 'count': vc_orig.values.tolist()})
            original_counts['Dataset'] = 'Original'

            vc_clean = clean_df[comp_col].value_counts().head(15)
            clean_counts = pd.DataFrame({comp_col: vc_clean.index.tolist(), 'count': vc_clean.values.tolist()})
            clean_counts['Dataset'] = 'Clean'

            combined = pd.concat([original_counts, clean_counts])
            fig_compare = px.bar(combined, x=comp_col, y='count',
                                 color='Dataset',
                                 barmode='group',
                                 color_discrete_map={'Original':'#f97316','Clean':'#22c55e'},
                                 title=f"Original vs Clean — {comp_col}",
                                 text='count')
            fig_compare.update_traces(textposition='outside')
            fig_compare.update_layout(**LAYOUT, height=420, xaxis_tickangle=-35)
            st.plotly_chart(fig_compare, use_container_width=True)

        # Footer note
        st.markdown(
            "<p style='color:#94a3b8;font-size:0.8rem;text-align:center;'>📊 Charts auto-generated from your dataset columns</p>",
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════════
# TAB — DATA PREVIEW
# ═══════════════════════════════════════════════════════════
elif active == "📊 Data Preview":
    st.title("📊 Raw Data Preview")
    st.caption("Full view of the uploaded dataset.")
    st.divider()
    if st.session_state.df is None:
        st.warning("⚠️ No file loaded yet. Go to **📂 Upload** first.")
    else:
        df = st.session_state.df
        st.info(f"Showing {len(df)} records  ·  {len(df.columns)} columns")
        st.dataframe(df, use_container_width=True, height=600)

# ═══════════════════════════════════════════════════════════
# TAB — DUPLICATES
# ═══════════════════════════════════════════════════════════
elif active == "🔍 Duplicates":
    st.title("🔍 Duplicate Explorer")
    st.caption("Browse and compare duplicate record groups.")
    st.divider()
    if st.session_state.df is None:
        st.warning("⚠️ No file loaded yet. Go to **📂 Upload** first.")
    else:
        df         = st.session_state.df
        duplicates = st.session_state.duplicates
        text_col   = st.session_state.text_col

        c1, c2, c3 = st.columns(3)
        c1.metric("⚡ Embedding Time", f"{st.session_state.embed_time:.2f}s")
        c2.metric("🔍 Detection Time", f"{st.session_state.detect_time:.2f}s")
        c3.metric("⏱️ Total Time",      f"{st.session_state.total_time:.2f}s")

        st.divider()
        st.subheader(f"🔁 Duplicate Records ({len(duplicates)} rows)")
        st.dataframe(duplicates.head(1000), use_container_width=True, height=350)

        st.divider()
        st.subheader("🔎 Group Explorer")
        if len(duplicates) > 0:
            group_ids      = sorted(duplicates['group'].unique())
            selected_group = st.selectbox("Select a duplicate group", group_ids)
            group_data     = duplicates[duplicates['group'] == selected_group]
            st.write(f"**Group {selected_group}** — {len(group_data)} items")
            st.dataframe(group_data, use_container_width=True)
            st.write("#### 🧠 Pairwise Comparison")
            texts_in_group = group_data[text_col].tolist()
            for i in range(min(3, len(texts_in_group))):
                for j in range(i + 1, min(3, len(texts_in_group))):
                    st.markdown(f"> `{texts_in_group[i]}`  ↔  `{texts_in_group[j]}`")
        else:
            st.info("No duplicate groups found.")

# ═══════════════════════════════════════════════════════════
# TAB — CLEAN DATASET
# ═══════════════════════════════════════════════════════════
elif active == "🧹 Clean Dataset":
    st.title("🧹 Clean Dataset")
    st.caption("Deduplicated dataset — one record per semantic group.")
    st.divider()
    if st.session_state.clean_df is None:
        st.warning("⚠️ No file loaded yet. Go to **📂 Upload** first.")
    else:
        clean_df = st.session_state.clean_df
        df       = st.session_state.df
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Original Records", len(df))
        c2.metric("🧹 Clean Records",    len(clean_df))
        c3.metric("🗑️ Removed",          len(df) - len(clean_df))
        st.divider()
        st.dataframe(clean_df, use_container_width=True, height=500)
        csv_clean = clean_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Clean Dataset", csv_clean,
                           file_name="clean_dataset.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════
# TAB — TRANSLATE
# ═══════════════════════════════════════════════════════════
elif active == "🌐 Translate":
    st.title("🌐 Translate Clean Dataset")
    st.caption("Translate selected columns into your preferred language.")
    st.divider()
    if st.session_state.clean_df is None:
        st.warning("⚠️ No file loaded yet. Go to **📂 Upload** first.")
    else:
        clean_df = st.session_state.clean_df
        text_col = st.session_state.text_col

        lang_map = {
            "English": "en", "Hindi": "hi", "Japanese": "ja",
            "German":  "de", "French": "fr", "Spanish":  "es",
        }

        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("⚙️ Settings")
            lang_options   = list(lang_map.keys())
            selected_lang  = st.selectbox("🌍 Target language", lang_options,
                                          index=lang_options.index("English"))
            all_text_cols  = [c for c in clean_df.columns if clean_df[c].dtype == object]
            cols_to_trans  = st.multiselect(
                "📋 Columns to translate", options=all_text_cols,
                default=[text_col] if text_col in all_text_cols else all_text_cols[:1],
                help="Each column gets its own translated_<column> output"
            )
            total_clean = len(clean_df)
            max_rows    = st.slider("🔢 Row limit", 1, total_clean,
                                    min(300, total_clean), step=1)
            st.info(f"**{max_rows}** rows × **{len(cols_to_trans)}** column(s)")
            translate_btn = st.button("🌐 Translate Now", use_container_width=True)

        with col_right:
            st.subheader("📋 Preview (Clean Data)")
            st.dataframe(clean_df.head(10), use_container_width=True)

        if translate_btn:
            if not cols_to_trans:
                st.warning("⚠️ Select at least one column.")
            else:
                st.divider()
                trans_start       = time.time()
                trans_df          = clean_df.head(max_rows).copy()
                total_to_trans    = len(trans_df)
                target_code       = lang_map[selected_lang]
                translation_cache = {}

                def translate_cell(text_str, tgt, cache):
                    key = (text_str, tgt)
                    if key in cache:
                        return cache[key]
                    clean_text = str(text_str).strip()[:4000]
                    if not clean_text:
                        return text_str
                    for engine in ['google', 'bing', 'alibaba']:
                        try:
                            import translators as ts
                            result = ts.translate_text(clean_text, translator=engine,
                                                       to_language=tgt, timeout=10)
                            if result and result.strip():
                                cache[key] = result
                                cache['_last_engine'] = engine
                                return result
                        except Exception:
                            time.sleep(0.3)
                    cache[key] = text_str
                    return text_str

                num_cols       = len(cols_to_trans)
                trans_progress = st.progress(0, text="Starting translation...")

                for col_idx, col in enumerate(cols_to_trans):
                    st.write(f"🔄 Translating **{col}** → `translated_{col}`")
                    results = []
                    for idx, t in enumerate(trans_df[col].astype(str).tolist()):
                        results.append(translate_cell(str(t), target_code, translation_cache))
                        overall = col_idx * total_to_trans + (idx + 1)
                        eng     = translation_cache.get('_last_engine', 'google').capitalize()
                        trans_progress.progress(
                            int(overall / (num_cols * total_to_trans) * 100),
                            text=f"[{eng}] '{col}': row {idx+1}/{total_to_trans}"
                        )
                    trans_df[f"translated_{col}"] = results

                trans_end = time.time()
                trans_progress.progress(100, text="✅ Complete!")
                st.success(
                    f"✅ {total_to_trans} rows × {num_cols} column(s) → "
                    f"**{selected_lang}** in {trans_end - trans_start:.2f}s"
                )
                st.dataframe(trans_df, use_container_width=True)
                csv_trans = trans_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"📥 Download Translated Dataset ({selected_lang})", csv_trans,
                    file_name=f"translated_{selected_lang.lower()}.csv", mime="text/csv"
                )
