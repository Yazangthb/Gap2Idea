"""Gap2Idea — clean, opinionated dashboard.

Layout: sidebar navigation, one task per page, big primary CTAs.
"""
from __future__ import annotations

import json
import logging
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    from gap2idea.pipeline.openai_ideas import (
        _diverse_evidence,
        _evidence_payload,
        _retrieve_methods_for_cluster,
        generate_idea_for_pair,
    )
    from gap2idea.pipeline.semantic_scholar import S2Client
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    logger.exception(
        "Failed to import gap2idea; cwd=%s __file__=%s sys.path=%s",
        Path.cwd(), Path(__file__).resolve(), sys.path,
    )
    st.error("Couldn't import `gap2idea`. Run `pip install -e .` or `uv sync` first.")
    raise


# =====================================================================
# Constants & global styling
# =====================================================================

ART_DIR = Path("artifacts")
DATA_DIR = Path("data")
PDFS_DIR = DATA_DIR / "pdfs"

st.set_page_config(
    page_title="Gap2Idea",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Tighten top padding so the page starts higher */
    .main .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
    /* Sidebar polish */
    [data-testid="stSidebar"] {background: #f5f5f5;}
    [data-testid="stSidebar"] .stRadio > label {font-weight: 600; font-size: 0.95rem;}
    /* Card-ish containers */
    div[data-testid="stContainer"] {border-radius: 12px;}
    /* Bigger primary buttons */
    button[kind="primary"] {
        font-weight: 600; padding: 0.65rem 1.5rem; font-size: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }
    /* Metric labels: a bit larger */
    [data-testid="stMetricValue"] {font-size: 2rem; font-weight: 700;}
    [data-testid="stMetricLabel"] {font-size: 0.8rem; letter-spacing: 0.02em;
        text-transform: uppercase; color: #64748b;}
    /* Section headings */
    h1, h2, h3 {color: #0f172a; letter-spacing: -0.01em;}
    /* Smaller captions */
    .stCaption, [data-testid="stCaptionContainer"] {color: #64748b;}
    /* Hide the "Made with Streamlit" footer */
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# Cached loaders
# =====================================================================

@st.cache_data(show_spinner=False)
def load_tsv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, sep="\t")


@st.cache_resource
def load_embeddings(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return np.load(p)


@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def compute_2d(X):
    from sklearn.manifold import TSNE
    perp = min(30, max(2, len(X) // 4))
    return TSNE(
        n_components=2, random_state=42, perplexity=perp, max_iter=500
    ).fit_transform(X)


@st.cache_data(show_spinner=False)
def load_paper_meta() -> dict:
    p = ART_DIR / "papers_metadata.tsv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, sep="\t").fillna("")
    return {
        str(r["id"]): {
            "title": str(r.get("title", "")),
            "abstract": str(r.get("abstract", "")),
            "venue": str(r.get("venue", "")),
            "year": r.get("year", ""),
            "citation_count": r.get("citation_count", 0),
            "url": str(r.get("url", "")),
        }
        for _, r in df.iterrows()
    }


@st.cache_resource
def get_s2_client():
    return S2Client()


@st.cache_data(show_spinner=False)
def s2_search(query: str, limit: int = 10) -> list[dict]:
    if not query.strip():
        return []
    try:
        return get_s2_client().search(query, limit=limit)
    except Exception as e:
        st.warning(f"Semantic Scholar search failed: {e}")
        return []


def resolve_title(paper_id, paper_meta: dict) -> str:
    meta = paper_meta.get(str(paper_id))
    if meta and meta.get("title"):
        return meta["title"]
    pdf_path = PDFS_DIR / f"{paper_id}.pdf"
    if pdf_path.exists():
        try:
            import fitz
            doc = fitz.open(pdf_path)
            t = (doc.metadata or {}).get("title") or ""
            doc.close()
            if t:
                return t
        except Exception:
            pass
    return str(paper_id)


# =====================================================================
# Shared data load (runs once per session)
# =====================================================================

@st.cache_data(show_spinner=False)
def _load_all_data():
    return {
        "gaps":      load_tsv(str(ART_DIR / "gaps_with_clusters.tsv")),
        "clusters":  load_tsv(str(ART_DIR / "cluster_summary.tsv")),
        "pairs":     load_tsv(str(ART_DIR / "cluster_pairs.tsv")),
        "labels":    load_tsv(str(ART_DIR / "cluster_labels.tsv")),
        "ideas":     load_tsv(str(ART_DIR / "ideas.tsv")),
        "idea_eval": load_tsv(str(ART_DIR / "idea_eval.tsv")),
    }


def empty_state(icon: str, title: str, body: str, cmd: str | None = None):
    """A friendly placeholder when an artifact is missing."""
    with st.container(border=True):
        st.markdown(f"### {icon}  {title}")
        st.write(body)
        if cmd:
            st.code(cmd, language="bash")


# =====================================================================
# Page: Home
# =====================================================================

def page_home(data):
    gaps, clusters, ideas, idea_eval, pairs = (
        data["gaps"], data["clusters"], data["ideas"],
        data["idea_eval"], data["pairs"],
    )

    st.title("Gap2Idea")
    st.caption(
        "Mine research gaps from papers → cluster them into themes → "
        "synthesise novel, evidence-grounded research ideas."
    )
    st.divider()

    if gaps.empty:
        empty_state(
            "🚧", "No corpus loaded yet",
            "Run the upstream pipeline to populate `artifacts/`. See the README for details.",
            "gap2idea run-all",
        )
        return

    # ---- Top metrics ----
    n_papers = gaps["id"].nunique()
    n_gaps = len(gaps)
    real_clusters = [c for c in gaps["cluster_id"].unique() if c != -1]
    n_themes = len(real_clusters)
    n_ideas = len(ideas)
    n_evals = len(idea_eval)
    composite = (
        f"{pd.to_numeric(idea_eval['composite'], errors='coerce').mean():.2f} / 5"
        if not idea_eval.empty and "composite" in idea_eval.columns else "—"
    )

    cols = st.columns(5)
    cols[0].metric("Papers", n_papers)
    cols[1].metric("Gaps mined", n_gaps)
    cols[2].metric("Themes", n_themes)
    cols[3].metric("Ideas", n_ideas)
    cols[4].metric("Avg quality", composite, help="Mean composite score from the LLM judge")

    st.divider()

    # ---- Pipeline status ----
    st.subheader("Pipeline status")
    rows = [
        ("📄 Papers → text",  not gaps.empty, "gap2idea extract-text"),
        ("✂️  Sections",      not gaps.empty, "gap2idea extract-sections"),
        ("🔍 Gaps extracted", not gaps.empty, "gap2idea extract-gaps"),
        ("🧩 Themes clustered", not clusters.empty, "gap2idea theme-mine"),
        ("🌉 Bridges scored", not pairs.empty, "gap2idea theme-mine (--top-pairs)"),
        ("💡 Ideas generated", not ideas.empty, "gap2idea generate-ideas"),
        ("📊 Ideas evaluated", not idea_eval.empty, "gap2idea evaluate-ideas"),
    ]
    for label, done, cmd in rows:
        c1, c2, c3 = st.columns([3, 1, 5])
        c1.write(label)
        c2.write("✅" if done else "⚪")
        if done:
            c3.caption("done")
        else:
            c3.code(cmd, language="bash")

    st.divider()

    # ---- Action cards ----
    st.subheader("Where to next?")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("### 🎯  Explore themes")
        st.caption(f"{n_themes} clusters · {n_gaps} gap statements")
        st.write("See which research opportunities recur across the corpus.")
        if st.button("Open themes →", width="stretch", key="home_to_themes"):
            st.session_state["page"] = "Themes"
            st.rerun()
    with c2, st.container(border=True):
        st.markdown("### ✨  Generate an idea")
        st.caption("3 modes: bridge · within · method-gap")
        st.write("Synthesise a new research idea on demand and check it against prior work.")
        if st.button("Open generator →", width="stretch", key="home_to_gen"):
            st.session_state["page"] = "Generate"
            st.rerun()
    with c3, st.container(border=True):
        st.markdown("### 💡  Idea library")
        st.caption(f"{n_ideas} saved · {n_evals} judged")
        st.write("Browse, filter and export all generated ideas.")
        if st.button("Open library →", width="stretch", key="home_to_ideas"):
            st.session_state["page"] = "Ideas"
            st.rerun()


# =====================================================================
# Page: Themes
# =====================================================================

def page_themes(data, paper_meta, embeddings, label_map):
    gaps, clusters, labels = data["gaps"], data["clusters"], data["labels"]

    st.title("🎯  Themes")
    st.caption("Recurring research opportunities across the corpus.")

    if clusters.empty:
        empty_state("🧩", "No themes yet", "Run theme-mine to embed and cluster gaps.",
                    "gap2idea theme-mine")
        return

    # ---- Cluster picker (large, prominent) ----
    opts = clusters["cluster_id"].astype(int).tolist()
    default = st.session_state.get("selected_cluster", opts[0])
    if default not in opts:
        default = opts[0]
    idx = opts.index(default)
    selected = st.selectbox(
        "Choose a theme",
        opts, index=idx,
        format_func=lambda c: f"  {label_map.get(int(c), '?')}",
        label_visibility="visible",
    )
    st.session_state["selected_cluster"] = int(selected)

    crow = clusters[clusters["cluster_id"] == selected].iloc[0]

    # ---- Theme summary card ----
    with st.container(border=True):
        st.markdown(f"## {crow['theme_label']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Gaps in this theme", int(crow["n_items"]))
        c2.metric("Papers", int(crow["n_papers"]))
        c3.metric("Avg gap confidence", f"{float(crow['avg_conf']):.2f}")
        if not labels.empty:
            kw = labels[labels["cluster_id"] == selected]
            if not kw.empty and "keywords" in kw.columns:
                st.caption(f"**Keywords**  ·  {kw.iloc[0]['keywords']}")

    st.markdown("### Source papers & their gap statements")

    sub = gaps[gaps["cluster_id"] == selected].sort_values("confidence", ascending=False)
    # group by paper, list gap sentences underneath
    for pid, grp in sub.groupby("id"):
        meta = paper_meta.get(str(pid), {})
        title = resolve_title(pid, paper_meta)
        with st.container(border=True):
            st.markdown(f"**{title}**")
            bits = []
            if meta.get("venue"): bits.append(meta["venue"])
            if meta.get("year"):  bits.append(str(meta["year"]))
            if meta.get("citation_count") not in ("", None):
                bits.append(f"{meta['citation_count']} citations")
            if bits:
                st.caption(" · ".join(bits))
            for _, g in grp.iterrows():
                badge = "🔴" if g["confidence"] >= 0.9 else "🟡"
                st.markdown(f"{badge}  {g['gap_sentence']}")
            cols = st.columns(3)
            cols[0].markdown(f"[arXiv ↗](https://arxiv.org/abs/{pid})")
            if meta.get("url"):
                cols[1].markdown(f"[Semantic Scholar ↗]({meta['url']})")
            pdf = PDFS_DIR / f"{pid}.pdf"
            if pdf.exists():
                cols[2].download_button(
                    "📄 PDF", pdf.read_bytes(), file_name=f"{pid}.pdf",
                    mime="application/pdf", key=f"pdf_{pid}_{selected}",
                    width="stretch",
                )


# =====================================================================
# Page: Bridges
# =====================================================================

def page_bridges(data, label_map):
    gaps, pairs = data["gaps"], data["pairs"]

    st.title("🌉  Bridges")
    st.caption(
        "Pairs of themes that sit in the *bridge sweet spot* — similar enough "
        "to be combined into one idea, distinct enough that the combination is novel."
    )

    if pairs.empty:
        empty_state("🌉", "No bridges yet", "Need at least 2 themes. Re-run theme-mine.",
                    "gap2idea theme-mine --top-pairs 20")
        return

    display = pairs.copy()
    if "bridge_score" in display.columns:
        display = display.sort_values("bridge_score", ascending=False)

    # Compact, readable table — only the columns that matter
    show_cols = [c for c in ["label_a", "label_b", "bridge_score", "cosine_sim",
                              "paper_overlap", "type_complementarity"] if c in display.columns]
    nice = display[show_cols].rename(columns={
        "label_a": "Theme A", "label_b": "Theme B",
        "bridge_score": "Bridge", "cosine_sim": "Similarity",
        "paper_overlap": "Paper overlap", "type_complementarity": "Type diversity",
    })
    st.dataframe(
        nice.head(20),
        width="stretch", hide_index=True, height=420,
        column_config={
            "Bridge": st.column_config.ProgressColumn(
                "Bridge", help="Composite bridge score, higher is better",
                min_value=0.0, max_value=1.0, format="%.2f",
            ),
            "Similarity": st.column_config.NumberColumn(format="%.2f"),
            "Paper overlap": st.column_config.NumberColumn(format="%.2f"),
            "Type diversity": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.markdown("### Inspect a bridge")
    idx = st.slider("Pick a row", 0, max(0, len(display) - 1),
                    st.session_state.get("selected_pair", 0), key="bridge_slider")
    st.session_state["selected_pair"] = idx
    pr = display.iloc[idx]
    ca, cb = int(pr["cluster_a"]), int(pr["cluster_b"])

    col_a, col_b = st.columns(2)
    for col, c_id, label_key in [(col_a, ca, "label_a"), (col_b, cb, "label_b")]:
        with col, st.container(border=True):
            st.markdown(f"#### {pr.get(label_key, '')}")
            ev = gaps[gaps.cluster_id == c_id].sort_values("confidence", ascending=False).head(6)
            for _, e in ev.iterrows():
                st.markdown(f"- *({e['id']})*  {e['gap_sentence']}")

    st.divider()
    if st.button("✨  Generate idea from this bridge", type="primary", width="stretch"):
        with st.spinner("Calling LLM + checking novelty…"):
            idea = generate_idea_for_pair(
                gaps, ca, cb,
                label_a=str(pr.get("label_a", "")),
                label_b=str(pr.get("label_b", "")),
                check_novelty=True,
            )
        if idea:
            _save_idea(idea)
            st.success("Saved to your idea library.")
            st.session_state["last_idea"] = idea
            _show_idea_card(idea)
        else:
            st.error("Couldn't generate (empty evidence?).")


# =====================================================================
# Page: Generate
# =====================================================================

def _save_idea(idea: dict):
    existing = load_tsv(str(ART_DIR / "ideas.tsv"))
    new = pd.DataFrame([idea])
    merged = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    ART_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(ART_DIR / "ideas.tsv", sep="\t", index=False)
    load_tsv.clear()
    _load_all_data.clear()


def _show_idea_card(idea: dict):
    with st.container(border=True):
        st.markdown(f"### {idea['title']}")
        nov = idea.get("novelty_score")
        conf = idea.get("idea_confidence")
        bits = []
        if conf is not None:
            bits.append(f"**Confidence** {float(conf):.0%}")
        if nov is not None and not pd.isna(nov):
            bits.append(f"**Novelty** {float(nov):.0%}")
        if bits:
            st.caption("  ·  ".join(bits))

        st.markdown(f"**Research question.**  {idea['research_question']}")
        st.markdown(f"**Method.**  {idea['method_sketch']}")
        st.markdown(f"**Evaluation.**  {idea['evaluation_plan']}")
        st.markdown(f"**Contribution.**  {idea['expected_contribution']}")
        st.markdown(f"**Risks.**  {idea['assumptions_and_risks']}")
        if idea.get("closest_paper_title"):
            st.caption(
                f"**Closest prior work.**  *{idea['closest_paper_title']}*  "
                f"({idea.get('closest_paper_year', '?')})"
            )


def page_generate(data, embeddings, label_map):
    gaps, clusters, ideas = data["gaps"], data["clusters"], data["ideas"]

    st.title("✨  Generate an idea")
    st.caption("Three modes, each grounded in evidence from your corpus.")

    if clusters.empty:
        empty_state("🧩", "No themes available", "Run theme-mine first.",
                    "gap2idea theme-mine")
        return

    # ---- Mode picker (visual radio) ----
    mode_descriptions = {
        "Within one theme":
            "Synthesise one idea from the recurring gap statements inside a single cluster. "
            "Simplest and most defensible. Use this first.",
        "Bridge two themes":
            "Combine two related-but-distinct themes. "
            "Creative recombination — best when both themes have substantive evidence.",
        "Apply a method to a theme":
            "Retrieve method-claim sentences (from `data/methods.tsv`) that sit in the "
            "similarity sweet spot for a chosen theme, then ask the LLM to apply one. "
            "Most explicit *X-solves-Y* structure. Needs `gap2idea extract-methods` first.",
    }
    mode = st.radio("Mode", list(mode_descriptions.keys()), horizontal=True)
    st.caption(mode_descriptions[mode])
    st.divider()

    opts = clusters["cluster_id"].astype(int).tolist()
    fmt = lambda c: label_map.get(int(c), "?")

    if mode == "Bridge two themes":
        col_a, col_b = st.columns(2)
        ca = col_a.selectbox("Theme A", opts, format_func=fmt, key="gen_a")
        cb = col_b.selectbox("Theme B", [o for o in opts if o != ca], format_func=fmt, key="gen_b")
        k_evidence = st.slider("Evidence sentences per theme", 2, 8, 4)
        nov_chk = st.checkbox("Check novelty against Semantic Scholar", value=True)

        if st.button("Generate idea →", type="primary", width="stretch"):
            with st.spinner("Generating…"):
                idea = generate_idea_for_pair(
                    gaps, int(ca), int(cb),
                    label_a=label_map.get(int(ca), ""),
                    label_b=label_map.get(int(cb), ""),
                    check_novelty=nov_chk, k_evidence=k_evidence,
                )
            if idea:
                _save_idea(idea)
                st.session_state["last_idea"] = idea
                st.success("Saved.")
            else:
                st.error("No evidence available for one of those themes.")

    elif mode == "Within one theme":
        ca = st.selectbox("Theme", opts, format_func=fmt, key="gen_w")
        k_evidence = st.slider("Evidence sentences", 4, 12, 8)
        nov_chk = st.checkbox("Check novelty against Semantic Scholar", value=True)

        if st.button("Generate idea →", type="primary", width="stretch"):
            from gap2idea.pipeline.openai_ideas import (
                _build_within_prompt, _call_llm_within, novelty_check,
            )
            from gap2idea.pipeline.llm import get_llm_client

            with st.spinner("Generating…"):
                ev_df = _diverse_evidence(gaps, int(ca), k=k_evidence)
                if ev_df.empty:
                    st.error("No evidence for this theme.")
                else:
                    ev = _evidence_payload(ev_df)
                    prompt = _build_within_prompt(int(ca), label_map.get(int(ca), ""), ev)
                    client = get_llm_client()
                    data = _call_llm_within(client, prompt, model="openai/gpt-4.1-mini")
                    raw = data["idea"]
                    nov_payload = {}
                    if nov_chk:
                        try:
                            emb = load_embedder()
                            nov_payload = novelty_check(raw, get_s2_client(), emb, top_k=10)
                        except Exception as e:
                            st.warning(f"Novelty check failed: {e}")
                    closest = nov_payload.get("closest_paper") or {}
                    idea = {
                        "mode": "within",
                        "cluster_a": int(ca), "cluster_b": None,
                        "label_a": label_map.get(int(ca), ""), "label_b": "",
                        "title": raw["title"],
                        "research_question": raw["research_question"],
                        "method_sketch": raw["method_sketch"],
                        "evaluation_plan": raw["evaluation_plan"],
                        "expected_contribution": raw["expected_contribution"],
                        "assumptions_and_risks": raw["assumptions_and_risks"],
                        "idea_confidence": raw["confidence"],
                        "evidence_used_json": json.dumps(raw["evidence_used"], ensure_ascii=False),
                        "novelty_score": nov_payload.get("novelty_score"),
                        "max_similarity_to_prior": nov_payload.get("max_similarity"),
                        "closest_paper_title": closest.get("title", ""),
                        "closest_paper_year": closest.get("year", ""),
                        "closest_paper_id": closest.get("paperId", ""),
                    }
                    _save_idea(idea)
                    st.session_state["last_idea"] = idea
                    st.success("Saved.")

    else:  # Apply a method to a theme
        methods_path = DATA_DIR / "methods.tsv"
        if not methods_path.exists() or embeddings is None:
            empty_state(
                "🧪", "Method library not built yet",
                "Run `extract-methods` to mine method-claim sentences from the abstracts.",
                "gap2idea extract-methods",
            )
            return
        methods_all = pd.read_csv(methods_path, sep="\t")
        meth_emb_path = ART_DIR / "method_embeddings.npy"
        if meth_emb_path.exists() and len(np.load(meth_emb_path)) == len(methods_all):
            method_embeddings = np.load(meth_emb_path)
        else:
            with st.spinner("Embedding methods (first time only)…"):
                from gap2idea.pipeline.theme_mining import embed_sentences
                method_embeddings = embed_sentences(methods_all["method_sentence"].tolist())
                np.save(meth_emb_path, method_embeddings)

        ca = st.selectbox("Apply methods to which theme?", opts, format_func=fmt, key="gen_mg")
        c1, c2, c3 = st.columns(3)
        sim_low  = c1.slider("Similarity floor",   0.0, 1.0, 0.30, 0.05)
        sim_high = c2.slider("Similarity ceiling", 0.0, 1.0, 0.70, 0.05)
        k_meth   = c3.slider("Candidate methods", 1, 10, 5)

        retrieved = _retrieve_methods_for_cluster(
            int(ca), embeddings, gaps, methods_all, method_embeddings,
            top_k=k_meth, sim_low=sim_low, sim_high=sim_high,
        )

        if retrieved.empty:
            st.info("No methods fall in this similarity range. Widen the slider.")
        else:
            st.markdown(f"**Retrieved {len(retrieved)} candidate method(s):**")
            for _, m in retrieved.iterrows():
                with st.container(border=True):
                    badge = "🟢" if 0.4 <= m["_sim_to_gap_cluster"] <= 0.6 else "🟡"
                    st.markdown(
                        f"{badge} **{m['method_sentence']}**"
                    )
                    st.caption(
                        f"From paper `{m['id']}`  ·  "
                        f"type: {m.get('method_type', '?')}  ·  "
                        f"similarity to theme: {m['_sim_to_gap_cluster']:.2f}"
                    )

            k_evidence = st.slider("Gap evidence per theme", 2, 8, 4, key="mg_k_ev")
            nov_chk = st.checkbox("Check novelty against Semantic Scholar", value=True, key="mg_nov")

            if st.button("Generate idea →", type="primary", width="stretch"):
                from gap2idea.pipeline.openai_ideas import (
                    _build_method_gap_prompt, _call_llm_method_gap, novelty_check,
                )
                from gap2idea.pipeline.llm import get_llm_client

                with st.spinner("Generating…"):
                    gap_ev = _evidence_payload(_diverse_evidence(gaps, int(ca), k=k_evidence))
                    method_payload = [
                        {
                            "paper_id": str(r["id"]),
                            "method_type": str(r.get("method_type", "")),
                            "method_sentence": str(r["method_sentence"]),
                            "paragraph_text": str(r.get("paragraph_text", "")),
                            "similarity_to_gap_cluster": float(r["_sim_to_gap_cluster"]),
                        }
                        for _, r in retrieved.iterrows()
                    ]
                    prompt = _build_method_gap_prompt(
                        int(ca), label_map.get(int(ca), ""), gap_ev, method_payload,
                    )
                    client = get_llm_client()
                    data = _call_llm_method_gap(client, prompt, model="openai/gpt-4.1-mini")
                    raw = data["idea"]
                    nov_payload = {}
                    if nov_chk:
                        try:
                            emb = load_embedder()
                            nov_payload = novelty_check(raw, get_s2_client(), emb, top_k=10)
                        except Exception as e:
                            st.warning(f"Novelty check failed: {e}")
                    closest = nov_payload.get("closest_paper") or {}
                    idea = {
                        "mode": "method-gap",
                        "cluster_a": int(ca), "cluster_b": None,
                        "label_a": label_map.get(int(ca), ""), "label_b": "",
                        "title": raw["title"],
                        "research_question": raw["research_question"],
                        "method_sketch": raw["method_sketch"],
                        "evaluation_plan": raw["evaluation_plan"],
                        "expected_contribution": raw["expected_contribution"],
                        "assumptions_and_risks": raw["assumptions_and_risks"],
                        "idea_confidence": raw["confidence"],
                        "evidence_used_json": json.dumps(raw["evidence_used"], ensure_ascii=False),
                        "novelty_score": nov_payload.get("novelty_score"),
                        "max_similarity_to_prior": nov_payload.get("max_similarity"),
                        "closest_paper_title": closest.get("title", ""),
                        "closest_paper_year": closest.get("year", ""),
                        "closest_paper_id": closest.get("paperId", ""),
                    }
                    _save_idea(idea)
                    st.session_state["last_idea"] = idea
                    st.success("Saved.")

    # Show the most recent idea inline
    last = st.session_state.get("last_idea")
    if last:
        st.divider()
        st.subheader("Result")
        _show_idea_card(last)


# =====================================================================
# Page: Ideas library
# =====================================================================

def page_ideas(data):
    from gap2idea.pipeline.export import (
        BUNDLED_TEMPLATES,
        TEMPLATE_VARIABLES,
        find_latex_compiler,
    )

    ideas = data["ideas"]
    st.title("💡  Idea library")
    st.caption("Every generated idea, filtered and exportable.")

    if ideas.empty:
        empty_state(
            "💡", "No ideas saved yet",
            "Head to **Generate** to make one, or batch-generate from the CLI.",
            "gap2idea generate-ideas --mode within --n-pairs 5",
        )
        return

    # ---- LaTeX export controls (shared across all idea cards) ----
    with st.container(border=True):
        st.markdown("### 📄  LaTeX / PDF export")
        ec1, ec2 = st.columns([1, 2])
        with ec1:
            template_choice = st.selectbox(
                "Template",
                options=list(BUNDLED_TEMPLATES.keys()) + ["Custom (upload)"],
                index=list(BUNDLED_TEMPLATES.keys()).index("standard"),
                help="Bundled templates compile with stock TeX Live or tectonic.",
                key="latex_template",
            )
            compiler = find_latex_compiler()
            if compiler:
                st.success(f"LaTeX compiler found: **{compiler}**. PDF rendering enabled.")
            else:
                st.info(
                    "No LaTeX compiler on PATH — only `.tex` download available. "
                    "Install [tectonic](https://tectonic-typesetting.github.io/) "
                    "or TeX Live for one-click PDF."
                )
        with ec2:
            custom_template: str | None = None
            if template_choice == "Custom (upload)":
                up = st.file_uploader(
                    "Upload a Jinja2 LaTeX template (.tex.j2 / .tex)",
                    type=["j2", "tex"],
                    accept_multiple_files=False,
                    key="custom_tmpl",
                )
                if up is not None:
                    custom_template = up.getvalue().decode("utf-8", errors="replace")
                    st.caption(f"Loaded `{up.name}` ({len(custom_template)} chars). ✓")
                with st.expander("Template variables you can use"):
                    for k, v in TEMPLATE_VARIABLES.items():
                        st.markdown(f"- `{{{{ {k} }}}}` — {v}")
            else:
                st.caption(
                    "Selected template will be used for the per-idea download "
                    "and the bulk *Render to PDF* button below."
                )

    # Persist the resolved template + source on session_state so all cards see it.
    st.session_state["latex_template_name"] = (
        template_choice if template_choice != "Custom (upload)" else "standard"
    )
    st.session_state["latex_template_source"] = custom_template

    # ---- Filters in a tidy row ----
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.3])
        min_conf = c1.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
        min_nov  = c2.slider("Min novelty", 0.0, 1.0, 0.0, 0.05)
        mode_filter = c3.multiselect(
            "Modes",
            options=sorted(ideas["mode"].dropna().unique().tolist()) if "mode" in ideas.columns else [],
            default=sorted(ideas["mode"].dropna().unique().tolist()) if "mode" in ideas.columns else [],
        )
        sort_keys = [c for c in ["idea_confidence", "novelty_score", "bridge_score", "title"]
                     if c in ideas.columns]
        sort_by = c4.selectbox("Sort by", sort_keys, index=0 if sort_keys else 0)

    view = ideas.copy()
    if "idea_confidence" in view.columns:
        view = view[pd.to_numeric(view["idea_confidence"], errors="coerce").fillna(0) >= min_conf]
    if "novelty_score" in view.columns:
        view = view[pd.to_numeric(view["novelty_score"], errors="coerce").fillna(0) >= min_nov]
    if mode_filter and "mode" in view.columns:
        view = view[view["mode"].isin(mode_filter)]
    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=False, na_position="last")

    st.caption(f"Showing {len(view)} of {len(ideas)} ideas")

    for _, r in view.iterrows():
        with st.container(border=True):
            head_cols = st.columns([5, 1, 1, 1])
            head_cols[0].markdown(f"### {r['title']}")
            mode_label = r.get("mode", "")
            if mode_label:
                head_cols[1].markdown(f"`{mode_label}`")
            if "novelty_score" in r and pd.notna(r.get("novelty_score")):
                head_cols[2].metric("Novelty", f"{float(r['novelty_score']):.0%}", label_visibility="visible")
            if "idea_confidence" in r and pd.notna(r.get("idea_confidence")):
                head_cols[3].metric("Conf.", f"{float(r['idea_confidence']):.0%}", label_visibility="visible")
            if r.get("label_a") or r.get("label_b"):
                themes = f"**{r.get('label_a','')}**"
                if r.get("label_b"):
                    themes += f"  ✕  **{r.get('label_b','')}**"
                st.caption(themes)
            st.markdown(f"**RQ.**  {r.get('research_question','')}")
            with st.expander("Method · Evaluation · Risks", expanded=False):
                st.markdown(f"**Method sketch.**  {r.get('method_sketch','')}")
                st.markdown(f"**Evaluation.**  {r.get('evaluation_plan','')}")
                st.markdown(f"**Contribution.**  {r.get('expected_contribution','')}")
                st.markdown(f"**Risks.**  {r.get('assumptions_and_risks','')}")
                if r.get("closest_paper_title"):
                    st.caption(
                        f"Closest prior work: *{r['closest_paper_title']}* ({r.get('closest_paper_year','?')})"
                    )
                if "evidence_used_json" in r and pd.notna(r["evidence_used_json"]):
                    try:
                        ev = json.loads(r["evidence_used_json"])
                        st.caption(f"Grounded in {len(ev)} evidence quote(s)")
                        for e in ev:
                            st.markdown(f"- *({e.get('paper_id','?')})*  {e.get('gap_sentence','')}")
                    except Exception:
                        pass

            # Per-idea LaTeX + (optional) rendered-PDF export
            try:
                from gap2idea.pipeline.export import (
                    LatexCompilationError,
                    compile_latex_to_pdf,
                    find_latex_compiler,
                    idea_to_latex,
                )
                tmpl_name = st.session_state.get("latex_template_name", "standard")
                tmpl_src = st.session_state.get("latex_template_source")
                tex = idea_to_latex(
                    r.to_dict(),
                    template=tmpl_name, template_source=tmpl_src,
                )
                safe = "".join(c if c.isalnum() else "_" for c in str(r.get("title", "idea")))[:40] or "idea"
                row_uid = f"{r.get('title','')[:24]}_{int(r.get('cluster_a') or 0)}"

                btns = st.columns(2)
                btns[0].download_button(
                    "📄  Download .tex",
                    tex.encode("utf-8"),
                    file_name=f"{safe}.tex",
                    mime="application/x-tex",
                    key=f"tex_{row_uid}",
                    width="stretch",
                )
                if find_latex_compiler():
                    if btns[1].button("📑  Render & download PDF", key=f"renderpdf_{row_uid}", width="stretch"):
                        with st.spinner("Compiling LaTeX → PDF…"):
                            try:
                                pdf_bytes = compile_latex_to_pdf(tex)
                                st.session_state[f"_pdf_{row_uid}"] = pdf_bytes
                                st.success("Compiled. Click below to download.")
                            except LatexCompilationError as e:
                                st.error("LaTeX compile failed. Excerpt:")
                                st.code(str(e)[-1000:])
                    if f"_pdf_{row_uid}" in st.session_state:
                        st.download_button(
                            "⬇️  Download .pdf",
                            st.session_state[f"_pdf_{row_uid}"],
                            file_name=f"{safe}.pdf",
                            mime="application/pdf",
                            key=f"pdfdl_{row_uid}",
                            width="stretch",
                        )
                else:
                    btns[1].button("📑  Render & download PDF", disabled=True,
                                   help="Install tectonic or pdflatex to enable.",
                                   key=f"renderpdf_disabled_{row_uid}",
                                   width="stretch")
            except Exception as e:
                st.caption(f"LaTeX export unavailable: {e}")

    # Export
    st.divider()
    st.subheader("Bulk export")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "📊  Download as TSV",
        ideas.to_csv(sep="\t", index=False).encode("utf-8"),
        file_name="ideas.tsv", mime="text/tab-separated-values",
        width="stretch",
    )

    def to_md(df: pd.DataFrame) -> str:
        out = ["# Gap2Idea — generated ideas", ""]
        for _, r in df.iterrows():
            out += [
                f"## {r['title']}",
                f"**Mode:** {r.get('mode','—')}  ·  **Themes:** {r.get('label_a','')}"
                + (f" × {r.get('label_b','')}" if r.get('label_b') else ""),
                "",
                f"**Research question.** {r.get('research_question','')}", "",
                f"**Method sketch.** {r.get('method_sketch','')}", "",
                f"**Evaluation plan.** {r.get('evaluation_plan','')}", "",
                f"**Expected contribution.** {r.get('expected_contribution','')}", "",
                f"**Assumptions & risks.** {r.get('assumptions_and_risks','')}", "",
                "---", "",
            ]
        return "\n".join(out)

    c2.download_button(
        "📝  Download as Markdown",
        to_md(view).encode("utf-8"),
        file_name="ideas.md", mime="text/markdown",
        width="stretch",
    )

    # PDF library export
    try:
        from gap2idea.pipeline.export import ideas_to_pdf_bytes
        pdf_bytes = ideas_to_pdf_bytes(view)
        c3.download_button(
            "📕  Download as PDF",
            pdf_bytes,
            file_name="idea_library.pdf",
            mime="application/pdf",
            width="stretch",
        )
    except Exception as e:
        c3.caption(f"PDF unavailable: {e}")


# =====================================================================
# Page: Quality (evaluation)
# =====================================================================

def page_quality(data):
    idea_eval = data["idea_eval"]
    st.title("📊  Quality")
    st.caption("LLM-as-judge rubric scores for every generated idea.")

    if idea_eval.empty:
        empty_state(
            "📊", "No evaluations yet",
            "After generating ideas, run the judge to score them on a 1-5 rubric.",
            "gap2idea evaluate-ideas --judge-model anthropic/claude-sonnet-4",
        )
        return

    axes = ["novelty", "specificity", "feasibility", "evidence_grounding", "composite"]
    means = {a: pd.to_numeric(idea_eval[a], errors="coerce").mean()
             for a in axes if a in idea_eval.columns}

    cols = st.columns(len(means))
    for (k, v), col in zip(means.items(), cols):
        col.metric(k.replace("_", " ").title(), f"{v:.2f}" if pd.notna(v) else "—",
                   help="Mean across all judged ideas. Scale 1-5.")

    st.divider()
    st.markdown("### Distribution of scores")
    long_df = idea_eval[[a for a in axes if a in idea_eval.columns]].melt(
        var_name="axis", value_name="score"
    )
    long_df["axis"] = long_df["axis"].str.replace("_", " ").str.title()
    fig = px.box(long_df, x="axis", y="score", points="all",
                 color="axis", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=20, b=20),
                      yaxis=dict(range=[0, 5.5]), xaxis_title="", yaxis_title="Score")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Detailed scores")
    show = idea_eval.sort_values("composite", ascending=False)
    nice = show.rename(columns={
        "title": "Title",
        "novelty": "Novelty", "specificity": "Specificity",
        "feasibility": "Feasibility", "evidence_grounding": "Grounding",
        "composite": "Composite", "evidence_overlap": "Overlap",
    })
    keep = [c for c in ["Title", "Novelty", "Specificity", "Feasibility",
                         "Grounding", "Composite", "Overlap"] if c in nice.columns]
    st.dataframe(
        nice[keep], width="stretch", hide_index=True, height=350,
        column_config={
            "Composite": st.column_config.ProgressColumn(
                "Composite", min_value=0.0, max_value=5.0, format="%.2f",
            ),
            "Overlap": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.download_button(
        "Download evaluation TSV",
        idea_eval.to_csv(sep="\t", index=False).encode("utf-8"),
        file_name="idea_eval.tsv", mime="text/tab-separated-values",
    )


# =====================================================================
# Page: Landscape  (kept for power users)
# =====================================================================

def page_landscape(data, embeddings, label_map):
    gaps, clusters = data["gaps"], data["clusters"]
    st.title("🗺️  Gap landscape")
    st.caption("Visualisations of the embedding space and theme relationships.")

    if embeddings is None or clusters.empty:
        empty_state("🗺️", "Run the pipeline first",
                    "Embeddings and clusters are missing.", "gap2idea theme-mine")
        return

    # 2D scatter
    X_2d = compute_2d(embeddings)
    plot_df = pd.DataFrame({
        "x": X_2d[:, 0], "y": X_2d[:, 1],
        "theme": gaps["cluster_id"].map(label_map).fillna("noise"),
        "Gap statement": gaps["gap_sentence"],
        "Paper": gaps["id"].astype(str),
    })
    st.markdown("### Each dot is one gap statement (t-SNE projection)")
    fig = px.scatter(
        plot_df, x="x", y="y", color="theme",
        hover_data={"Gap statement": True, "Paper": True, "x": False, "y": False},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20),
                      xaxis_title="", yaxis_title="",
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
    st.plotly_chart(fig, width="stretch")


# =====================================================================
# Navigation
# =====================================================================

PAGES = {
    "Home":        ("🏠", page_home,      "needs:none"),
    "Themes":      ("🎯", page_themes,    "needs:meta+emb+labels"),
    "Bridges":     ("🌉", page_bridges,   "needs:labels"),
    "Generate":    ("✨", page_generate,  "needs:emb+labels"),
    "Ideas":       ("💡", page_ideas,     "needs:none"),
    "Quality":     ("📊", page_quality,   "needs:none"),
    "Landscape":   ("🗺️", page_landscape,  "needs:emb+labels"),
}


def main():
    data = _load_all_data()
    paper_meta = load_paper_meta()
    embeddings = load_embeddings(str(ART_DIR / "gap_embeddings.npy"))
    labels = data["labels"]
    label_map = (
        dict(zip(labels["cluster_id"].astype(int), labels["theme_label"]))
        if not labels.empty else {}
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("# 💡 Gap2Idea")
        st.caption("Research gaps → research ideas")
        st.divider()

        current = st.session_state.get("page", "Home")
        for name, (icon, _fn, _needs) in PAGES.items():
            if st.button(
                f"{icon}   {name}",
                width="stretch",
                type="primary" if name == current else "secondary",
                key=f"nav_{name}",
            ):
                st.session_state["page"] = name
                st.rerun()

        st.divider()

        # Pipeline freshness indicator
        with st.expander("📦  Corpus", expanded=False):
            if data["gaps"].empty:
                st.warning("No data loaded.")
            else:
                st.write(f"**Papers:** {data['gaps']['id'].nunique()}")
                st.write(f"**Gaps:**   {len(data['gaps'])}")
                st.write(f"**Ideas:**  {len(data['ideas'])}")
                if st.button("🔄  Refresh data", width="stretch"):
                    st.cache_data.clear()
                    st.rerun()

        st.caption("Made with the Gap2Idea pipeline.")

    # ---- Page dispatch ----
    current = st.session_state.get("page", "Home")
    icon, fn, _needs = PAGES[current]
    if fn is page_home:
        fn(data)
    elif fn is page_themes:
        fn(data, paper_meta, embeddings, label_map)
    elif fn is page_bridges:
        fn(data, label_map)
    elif fn is page_generate:
        fn(data, embeddings, label_map)
    elif fn is page_ideas:
        fn(data)
    elif fn is page_quality:
        fn(data)
    elif fn is page_landscape:
        fn(data, embeddings, label_map)


# Streamlit always executes the script top-to-bottom, so just call main().
main()
