"""Streamlit dashboard for Gap2Idea.

Layout:
  Overview   — corpus stats, cluster sizes, similarity heatmap
  Themes     — explore each cluster, see top papers & gaps
  Bridges    — bridge-scored cluster pairs with evidence side-by-side
  Generate   — produce a new idea on demand
  Ideas      — gallery of saved ideas, filter by score, export
  Evaluate   — run LLM-as-judge over ideas_full.jsonl

State:
  - st.session_state['selected_cluster']  (int)
  - st.session_state['selected_pair']     (int row index into pairs)
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
import streamlit as st

# Defensive import — gives a clear error if the package isn't on the path
# (e.g. running outside an `pip install -e .` or `uv sync` environment, or
# in a Docker image that mis-sets PYTHONPATH).
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
        "Failed to import gap2idea; cwd=%s __file__=%s sys.path=%s project_root_contents=%s",
        Path.cwd(),
        Path(__file__).resolve(),
        sys.path,
        [p.name for p in project_root.iterdir()],
    )
    st.error(
        "Failed to import the internal package 'gap2idea'. "
        "Check Docker PYTHONPATH, or run `pip install -e .` / `uv sync` first."
    )
    raise

ART_DIR = Path("artifacts")
DATA_DIR = Path("data")
PDFS_DIR = DATA_DIR / "pdfs"


# =====================================================================
# Loaders (cached)
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
    return TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=500).fit_transform(X)


@st.cache_data
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
        st.warning(f"S2 search failed: {e}")
        return []


# =====================================================================
# Helpers
# =====================================================================

def resolve_title(paper_id, paper_meta: dict) -> str:
    """Best paper title: S2 metadata -> PDF embedded title -> arxiv id."""
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


def render_paper(paper_id: str, paper_meta: dict, gap_sentence: str | None = None):
    meta = paper_meta.get(str(paper_id), {})
    title = resolve_title(paper_id, paper_meta)
    with st.container(border=True):
        st.markdown(f"**{title}**")
        bits = []
        if meta.get("venue"):
            bits.append(meta["venue"])
        if meta.get("year"):
            bits.append(str(meta["year"]))
        if meta.get("citation_count") not in ("", None):
            bits.append(f"{meta.get('citation_count')} citations")
        if bits:
            st.caption(" · ".join(bits))
        if meta.get("abstract"):
            with st.expander("Abstract"):
                st.write(meta["abstract"])
        if gap_sentence:
            st.markdown(f"> {gap_sentence}")
        cols = st.columns(3)
        with cols[0]:
            if meta.get("url"):
                st.markdown(f"[Semantic Scholar]({meta['url']})")
        with cols[1]:
            st.markdown(f"[arXiv](https://arxiv.org/abs/{paper_id})")
        with cols[2]:
            pdf_path = PDFS_DIR / f"{paper_id}.pdf"
            if pdf_path.exists():
                st.download_button(
                    "PDF", pdf_path.read_bytes(),
                    file_name=f"{paper_id}.pdf", mime="application/pdf",
                    key=f"pdf_{paper_id}_{gap_sentence[:20] if gap_sentence else 'x'}",
                )


# =====================================================================
# Page setup + data load
# =====================================================================

st.set_page_config(page_title="Gap2Idea", layout="wide", page_icon="💡")
st.title("💡 Gap2Idea — research gaps → research ideas")
st.caption(
    "Mine limitations & future-work statements from papers, cluster them into "
    "themes, and synthesise novel ideas at the bridges between themes."
)

gaps = load_tsv(str(ART_DIR / "gaps_with_clusters.tsv"))
clusters = load_tsv(str(ART_DIR / "cluster_summary.tsv"))
pairs = load_tsv(str(ART_DIR / "cluster_pairs.tsv"))
labels = load_tsv(str(ART_DIR / "cluster_labels.tsv"))
ideas = load_tsv(str(ART_DIR / "ideas.tsv"))
idea_eval = load_tsv(str(ART_DIR / "idea_eval.tsv"))

if gaps.empty:
    st.error(
        "No `artifacts/gaps_with_clusters.tsv` found. Run "
        "`python -m gap2idea.cli theme-mine --gaps-tsv data/gaps.tsv` first."
    )
    st.stop()

paper_meta = load_paper_meta()
if not paper_meta:
    st.warning(
        "Missing `artifacts/papers_metadata.tsv` — run "
        "`python -m gap2idea.cli fetch-metadata` for rich paper titles/abstracts."
    )

embeddings = load_embeddings(str(ART_DIR / "gap_embeddings.npy"))
X_2d = compute_2d(embeddings) if embeddings is not None else None

label_map = (
    dict(zip(labels["cluster_id"].astype(int), labels["theme_label"]))
    if not labels.empty else {}
)

# Defaults for session state
if "selected_cluster" not in st.session_state and not clusters.empty:
    st.session_state.selected_cluster = int(clusters.iloc[0]["cluster_id"])
if "selected_pair" not in st.session_state:
    st.session_state.selected_pair = 0


tab_overview, tab_themes, tab_bridges, tab_gen, tab_ideas, tab_eval = st.tabs(
    ["📊 Overview", "🎯 Themes", "🔗 Bridges", "✨ Generate", "💡 Ideas", "📈 Evaluate"]
)


# =====================================================================
# Tab: Overview
# =====================================================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", gaps["id"].nunique())
    c2.metric("Gaps", len(gaps))
    c3.metric("Themes", int((gaps["cluster_id"] != -1).sum() and gaps["cluster_id"].nunique() - int((gaps["cluster_id"] == -1).any())))
    c4.metric("Ideas generated", len(ideas))

    st.subheader("Cluster sizes")
    if not clusters.empty:
        bar = px.bar(
            clusters.sort_values("n_papers", ascending=True),
            x="n_papers", y="theme_label", orientation="h",
            hover_data=["n_items", "avg_conf"],
        )
        bar.update_layout(height=max(300, 30 * len(clusters)))
        st.plotly_chart(bar, use_container_width=True)

    if X_2d is not None:
        st.subheader("Gap landscape (t-SNE)")
        plot_df = pd.DataFrame({
            "x": X_2d[:, 0], "y": X_2d[:, 1],
            "cluster_id": gaps["cluster_id"],
            "theme": gaps["cluster_id"].map(label_map).fillna("noise"),
            "sentence": gaps["gap_sentence"],
            "paper": gaps["id"].astype(str),
        })
        fig = px.scatter(
            plot_df, x="x", y="y", color="theme",
            hover_data={"sentence": True, "paper": True, "x": False, "y": False},
            custom_data=["cluster_id"],
        )
        sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
        if sel and "selection" in sel and sel["selection"]["points"]:
            cid = sel["selection"]["points"][0]["customdata"][0]
            st.session_state.selected_cluster = int(cid)
            st.toast(f"Selected cluster {cid}. Switch to **Themes** tab to explore.")

    # Similarity heatmap of cluster centroids
    if embeddings is not None and not clusters.empty:
        st.subheader("Theme similarity heatmap")
        cids = sorted(c for c in gaps["cluster_id"].unique() if c != -1)
        centroids = np.stack([
            embeddings[(gaps["cluster_id"].values == c)].mean(axis=0) for c in cids
        ])
        # cosine since rows are L2-normalised already
        S = centroids @ centroids.T
        names = [f"{c}: {label_map.get(int(c), '')[:30]}" for c in cids]
        heat = px.imshow(S, x=names, y=names, color_continuous_scale="Viridis", aspect="auto")
        st.plotly_chart(heat, use_container_width=True)


# =====================================================================
# Tab: Themes
# =====================================================================
with tab_themes:
    if clusters.empty:
        st.info("No clusters. Run `theme-mine` first.")
    else:
        # Search box (in-corpus title fuzzy match using S2 metadata)
        query = st.text_input("Search papers in your corpus by title", "")
        if query:
            scored = []
            for pid in gaps["id"].astype(str).unique():
                t = resolve_title(pid, paper_meta)
                s = SequenceMatcher(None, query.lower(), t.lower()).ratio()
                scored.append((s, pid, t))
            scored.sort(reverse=True)
            for s, pid, t in scored[:5]:
                row = gaps[gaps["id"].astype(str) == pid].iloc[0]
                if st.button(f"{t} (cluster {row['cluster_id']})", key=f"q_{pid}"):
                    st.session_state.selected_cluster = int(row["cluster_id"])

        opts = clusters["cluster_id"].astype(int).tolist()
        cur = st.session_state.get("selected_cluster", opts[0])
        idx = opts.index(cur) if cur in opts else 0
        sel = st.selectbox(
            "Choose a theme",
            opts, index=idx,
            format_func=lambda c: f"{c} — {label_map.get(int(c), '?')}",
            key="theme_selector",
        )
        st.session_state.selected_cluster = int(sel)

        crow = clusters[clusters["cluster_id"] == sel].iloc[0]
        st.subheader(crow["theme_label"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Gaps", int(crow["n_items"]))
        c2.metric("Papers", int(crow["n_papers"]))
        c3.metric("Avg confidence", f"{float(crow['avg_conf']):.2f}")

        if not labels.empty:
            kw = labels[labels["cluster_id"] == sel]
            if not kw.empty and "keywords" in kw.columns:
                st.caption(f"Keywords: _{kw.iloc[0]['keywords']}_")

        st.markdown("### Papers in this theme")
        sub = gaps[gaps["cluster_id"] == sel].sort_values("confidence", ascending=False)
        for _, r in sub.iterrows():
            render_paper(str(r["id"]), paper_meta, gap_sentence=r["gap_sentence"])


# =====================================================================
# Tab: Bridges
# =====================================================================
with tab_bridges:
    if pairs.empty:
        st.info("No cluster pairs. Run `theme-mine` first.")
    else:
        st.markdown(
            "Pairs are ranked by **bridge_score** = "
            "`peak(cos_sim, 0.45) × (1 - paper_overlap) × type_complementarity`. "
            "High bridge_score means *related but distinct* themes — the sweet "
            "spot for novel idea synthesis."
        )

        display = pairs.copy()
        if "bridge_score" in display.columns:
            display = display.sort_values("bridge_score", ascending=False)
        st.dataframe(display, use_container_width=True, height=300)

        idx = st.slider("Inspect pair #", 0, max(0, len(display) - 1), st.session_state.selected_pair)
        st.session_state.selected_pair = idx
        pr = display.iloc[idx]
        ca, cb = int(pr["cluster_a"]), int(pr["cluster_b"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("cosine", f"{float(pr.get('cosine_sim', 0)):.3f}")
        m2.metric("paper overlap", f"{float(pr.get('paper_overlap', 0)):.2f}")
        m3.metric("type compl.", f"{float(pr.get('type_complementarity', 0)):.2f}")
        m4.metric("bridge", f"{float(pr.get('bridge_score', 0)):.3f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"#### A · Cluster {ca}")
            st.markdown(f"**{pr.get('label_a','')}**")
            ev = gaps[gaps.cluster_id == ca].sort_values("confidence", ascending=False).head(8)
            for _, e in ev.iterrows():
                st.markdown(f"- ({e['id']}, conf {e['confidence']:.2f}) {e['gap_sentence']}")
        with col_b:
            st.markdown(f"#### B · Cluster {cb}")
            st.markdown(f"**{pr.get('label_b','')}**")
            ev = gaps[gaps.cluster_id == cb].sort_values("confidence", ascending=False).head(8)
            for _, e in ev.iterrows():
                st.markdown(f"- ({e['id']}, conf {e['confidence']:.2f}) {e['gap_sentence']}")

        st.divider()
        if st.button("Generate idea for this pair →", type="primary"):
            with st.spinner("Calling OpenAI + Semantic Scholar..."):
                idea = generate_idea_for_pair(
                    gaps, ca, cb,
                    label_a=str(pr.get("label_a", "")),
                    label_b=str(pr.get("label_b", "")),
                    check_novelty=True,
                )
            if idea:
                # persist
                new = pd.DataFrame([idea])
                merged = pd.concat([ideas, new], ignore_index=True) if not ideas.empty else new
                (ART_DIR).mkdir(parents=True, exist_ok=True)
                merged.to_csv(ART_DIR / "ideas.tsv", sep="\t", index=False)
                st.success("Saved to ideas.tsv. Switch to **Ideas** tab to view.")
                st.rerun()
            else:
                st.error("Idea generation returned nothing (empty evidence?).")


# =====================================================================
# Tab: Generate — three modes
# =====================================================================
with tab_gen:
    if clusters.empty:
        st.info("No clusters available.")
    else:
        mode = st.radio(
            "Generation mode",
            ["bridge (two themes)", "within (one theme)", "method-gap (apply methods to a theme)"],
            horizontal=True,
            help=(
                "bridge   — pair two related-but-distinct gap clusters.\n"
                "within   — synthesise one idea per cluster from all its gaps.\n"
                "method-gap — retrieve method-claim sentences from data/methods.tsv whose similarity to the chosen cluster centroid sits in the sweet spot, then apply them."
            ),
        )
        opts = clusters["cluster_id"].astype(int).tolist()

        # -- mode-specific UI --
        gen_kind = mode.split()[0]
        cluster_a = cluster_b = None
        method_df = pd.DataFrame()

        if gen_kind == "bridge":
            c1, c2 = st.columns(2)
            cluster_a = c1.selectbox(
                "Theme A", opts, key="gen_a",
                format_func=lambda c: f"{c} — {label_map.get(int(c),'?')}",
            )
            cluster_b = c2.selectbox(
                "Theme B", [o for o in opts if o != cluster_a], key="gen_b",
                format_func=lambda c: f"{c} — {label_map.get(int(c),'?')}",
            )
        else:
            cluster_a = st.selectbox(
                "Theme", opts, key="gen_single",
                format_func=lambda c: f"{c} — {label_map.get(int(c),'?')}",
            )
            if gen_kind == "method-gap":
                methods_path = DATA_DIR / "methods.tsv"
                meth_emb_path = ART_DIR / "method_embeddings.npy"
                if not methods_path.exists() or embeddings is None:
                    st.warning(
                        "Need both `data/methods.tsv` and `artifacts/gap_embeddings.npy`. "
                        "Run `gap2idea extract-methods` first."
                    )
                else:
                    methods_all = pd.read_csv(methods_path, sep="\t")
                    if meth_emb_path.exists() and len(np.load(meth_emb_path)) == len(methods_all):
                        method_embeddings = np.load(meth_emb_path)
                    else:
                        with st.spinner("Embedding methods (first time only)..."):
                            from gap2idea.pipeline.theme_mining import embed_sentences
                            method_embeddings = embed_sentences(methods_all["method_sentence"].tolist())
                            np.save(meth_emb_path, method_embeddings)

                    c1, c2, c3 = st.columns(3)
                    sim_low = c1.slider("sim low", 0.0, 1.0, 0.30, 0.05)
                    sim_high = c2.slider("sim high", 0.0, 1.0, 0.70, 0.05)
                    k_meth = c3.slider("# methods", 1, 10, 5)
                    method_df = _retrieve_methods_for_cluster(
                        int(cluster_a), embeddings, gaps, methods_all, method_embeddings,
                        top_k=k_meth, sim_low=sim_low, sim_high=sim_high,
                    )
                    if method_df.empty:
                        st.info("No methods fell in the sweet spot for this cluster.")
                    else:
                        st.caption(f"Retrieved {len(method_df)} candidate methods (similarity {sim_low:.2f}-{sim_high:.2f}):")
                        for _, m in method_df.iterrows():
                            with st.expander(f"({m['id']}, sim={m['_sim_to_gap_cluster']:.2f}) {m['method_sentence'][:80]}"):
                                st.write(m["method_sentence"])
                                if m.get("paragraph_text"):
                                    st.caption(m["paragraph_text"][:300])

        # -- common controls --
        k = st.slider("Gap evidence per cluster", 2, 10, 4)
        nov = st.checkbox("Run novelty check against Semantic Scholar", value=True)

        if st.button("Generate idea", type="primary"):
            idea = None
            with st.spinner("Generating..."):
                if gen_kind == "bridge":
                    idea = generate_idea_for_pair(
                        gaps, int(cluster_a), int(cluster_b),
                        label_a=label_map.get(int(cluster_a), ""),
                        label_b=label_map.get(int(cluster_b), ""),
                        check_novelty=nov, k_evidence=k,
                    )
                elif gen_kind == "within":
                    from gap2idea.pipeline.openai_ideas import (
                        SYSTEM_WITHIN, _build_within_prompt, _call_llm_within, novelty_check,
                    )
                    from gap2idea.pipeline.llm import get_llm_client
                    ev_df = _diverse_evidence(gaps, int(cluster_a), k=max(k, 6))
                    if ev_df.empty:
                        st.error("No evidence for this cluster.")
                    else:
                        ev = _evidence_payload(ev_df)
                        prompt = _build_within_prompt(int(cluster_a), label_map.get(int(cluster_a), ""), ev)
                        client = get_llm_client()
                        data = _call_llm_within(client, prompt, model="openai/gpt-4.1-mini")
                        raw = data["idea"]
                        nov_payload = {}
                        if nov:
                            try:
                                from sentence_transformers import SentenceTransformer
                                emb = SentenceTransformer("all-MiniLM-L6-v2")
                                nov_payload = novelty_check(raw, S2Client(), emb, top_k=10)
                            except Exception as e:
                                st.warning(f"Novelty check failed: {e}")
                        closest = nov_payload.get("closest_paper") or {}
                        idea = {
                            "mode": "within",
                            "cluster_a": int(cluster_a),
                            "label_a": label_map.get(int(cluster_a), ""),
                            "title": raw["title"],
                            "research_question": raw["research_question"],
                            "method_sketch": raw["method_sketch"],
                            "evaluation_plan": raw["evaluation_plan"],
                            "expected_contribution": raw["expected_contribution"],
                            "assumptions_and_risks": raw["assumptions_and_risks"],
                            "idea_confidence": raw["confidence"],
                            "novelty_score": nov_payload.get("novelty_score"),
                            "max_similarity_to_prior": nov_payload.get("max_similarity"),
                            "closest_paper_title": closest.get("title", ""),
                            "closest_paper_year": closest.get("year", ""),
                            "closest_paper_id": closest.get("paperId", ""),
                            "evidence_used_json": json.dumps(raw["evidence_used"], ensure_ascii=False),
                        }
                elif gen_kind == "method-gap":
                    if method_df.empty:
                        st.error("Pick a sim range that retrieves at least one method.")
                    else:
                        from gap2idea.pipeline.openai_ideas import (
                            SYSTEM_METHOD_GAP, _build_method_gap_prompt,
                            _call_llm_method_gap, novelty_check,
                        )
                        from gap2idea.pipeline.llm import get_llm_client
                        gap_ev = _evidence_payload(_diverse_evidence(gaps, int(cluster_a), k=k))
                        method_payload = [
                            {
                                "paper_id": str(r["id"]),
                                "method_type": str(r.get("method_type", "")),
                                "method_sentence": str(r["method_sentence"]),
                                "paragraph_text": str(r.get("paragraph_text", "")),
                                "similarity_to_gap_cluster": float(r["_sim_to_gap_cluster"]),
                            }
                            for _, r in method_df.iterrows()
                        ]
                        prompt = _build_method_gap_prompt(
                            int(cluster_a), label_map.get(int(cluster_a), ""),
                            gap_ev, method_payload,
                        )
                        client = get_llm_client()
                        data = _call_llm_method_gap(client, prompt, model="openai/gpt-4.1-mini")
                        raw = data["idea"]
                        nov_payload = {}
                        if nov:
                            try:
                                from sentence_transformers import SentenceTransformer
                                emb = SentenceTransformer("all-MiniLM-L6-v2")
                                nov_payload = novelty_check(raw, S2Client(), emb, top_k=10)
                            except Exception as e:
                                st.warning(f"Novelty check failed: {e}")
                        closest = nov_payload.get("closest_paper") or {}
                        idea = {
                            "mode": "method-gap",
                            "cluster_a": int(cluster_a),
                            "label_a": label_map.get(int(cluster_a), ""),
                            "title": raw["title"],
                            "research_question": raw["research_question"],
                            "method_sketch": raw["method_sketch"],
                            "evaluation_plan": raw["evaluation_plan"],
                            "expected_contribution": raw["expected_contribution"],
                            "assumptions_and_risks": raw["assumptions_and_risks"],
                            "idea_confidence": raw["confidence"],
                            "novelty_score": nov_payload.get("novelty_score"),
                            "max_similarity_to_prior": nov_payload.get("max_similarity"),
                            "closest_paper_title": closest.get("title", ""),
                            "closest_paper_year": closest.get("year", ""),
                            "closest_paper_id": closest.get("paperId", ""),
                            "evidence_used_json": json.dumps(raw["evidence_used"], ensure_ascii=False),
                        }
            if idea:
                st.session_state["last_idea"] = idea
                new = pd.DataFrame([idea])
                merged = pd.concat([ideas, new], ignore_index=True) if not ideas.empty else new
                merged.to_csv(ART_DIR / "ideas.tsv", sep="\t", index=False)
                st.success("Saved.")
        last = st.session_state.get("last_idea")
        if last:
            with st.container(border=True):
                st.markdown(f"### {last['title']}")
                st.markdown(f"**Research question.** {last['research_question']}")
                st.markdown(f"**Method sketch.** {last['method_sketch']}")
                st.markdown(f"**Evaluation plan.** {last['evaluation_plan']}")
                st.markdown(f"**Expected contribution.** {last['expected_contribution']}")
                st.markdown(f"**Assumptions & risks.** {last['assumptions_and_risks']}")
                cols = st.columns(2)
                cols[0].metric(
                    "Idea confidence (self-reported)",
                    f"{float(last.get('idea_confidence', 0)):.2f}",
                )
                ns = last.get("novelty_score")
                cols[1].metric("Novelty (1 − max sim. to S2 hits)", f"{ns:.2f}" if ns is not None else "—")
                if last.get("closest_paper_title"):
                    st.caption(
                        f"Closest prior work: *{last['closest_paper_title']}* "
                        f"({last.get('closest_paper_year','?')}) — "
                        f"sim {float(last.get('max_similarity_to_prior') or 0):.2f}"
                    )


# =====================================================================
# Tab: Ideas (gallery + filters + export)
# =====================================================================
with tab_ideas:
    if ideas.empty:
        st.info("No ideas saved yet. Generate some from the Bridges or Generate tabs, "
                "or run `python -m gap2idea.cli generate-ideas`.")
    else:
        st.markdown(f"**{len(ideas)} saved ideas.**")

        # Filters
        cols = st.columns(3)
        min_idea_conf = cols[0].slider("Min self-reported confidence", 0.0, 1.0, 0.0, 0.05)
        min_novelty = cols[1].slider("Min novelty score", 0.0, 1.0, 0.0, 0.05)
        sort_by = cols[2].selectbox(
            "Sort by", ["bridge_score", "novelty_score", "idea_confidence", "title"],
            index=0,
        )

        view = ideas.copy()
        if "idea_confidence" in view.columns:
            view = view[pd.to_numeric(view["idea_confidence"], errors="coerce").fillna(0) >= min_idea_conf]
        if "novelty_score" in view.columns:
            view = view[pd.to_numeric(view["novelty_score"], errors="coerce").fillna(0) >= min_novelty]
        if sort_by in view.columns:
            view = view.sort_values(sort_by, ascending=False)

        for _, r in view.iterrows():
            with st.container(border=True):
                st.markdown(f"### {r['title']}")
                bits = []
                if "label_a" in r and "label_b" in r:
                    bits.append(f"{r['label_a']} × {r['label_b']}")
                if "bridge_score" in r:
                    bits.append(f"bridge={float(r['bridge_score']):.2f}")
                if "idea_confidence" in r:
                    bits.append(f"self-conf={float(r['idea_confidence']):.2f}")
                if "novelty_score" in r and pd.notna(r.get("novelty_score")):
                    bits.append(f"novelty={float(r['novelty_score']):.2f}")
                st.caption(" · ".join(bits))
                st.markdown(f"**RQ.** {r['research_question']}")
                st.markdown(f"**Method.** {r['method_sketch']}")
                st.markdown(f"**Evaluation.** {r['evaluation_plan']}")
                st.markdown(f"**Contribution.** {r['expected_contribution']}")
                st.markdown(f"**Risks.** {r['assumptions_and_risks']}")
                if r.get("closest_paper_title"):
                    st.caption(
                        f"Closest prior work: *{r['closest_paper_title']}* ({r.get('closest_paper_year','?')})"
                    )
                # Evidence
                if "evidence_used_json" in r and pd.notna(r["evidence_used_json"]):
                    try:
                        ev = json.loads(r["evidence_used_json"])
                        with st.expander(f"{len(ev)} evidence quote(s)"):
                            for e in ev:
                                st.markdown(f"- ({e.get('paper_id','?')}) {e.get('gap_sentence','')}")
                    except Exception:
                        pass

        # Export
        st.divider()
        st.subheader("Export")
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download ideas.tsv", ideas.to_csv(sep="\t", index=False).encode("utf-8"),
            file_name="ideas.tsv", mime="text/tab-separated-values",
        )

        def to_md(df: pd.DataFrame) -> str:
            out = ["# Gap2Idea — generated ideas", ""]
            for _, r in df.iterrows():
                out += [
                    f"## {r['title']}",
                    f"**Themes:** {r.get('label_a','')} × {r.get('label_b','')}  ",
                    f"**Bridge score:** {r.get('bridge_score','')}  ",
                    f"**Self-confidence:** {r.get('idea_confidence','')}  ",
                    f"**Novelty (1 − max S2 sim):** {r.get('novelty_score','')}",
                    "",
                    f"**Research question.** {r.get('research_question','')}",
                    "",
                    f"**Method sketch.** {r.get('method_sketch','')}",
                    "",
                    f"**Evaluation plan.** {r.get('evaluation_plan','')}",
                    "",
                    f"**Expected contribution.** {r.get('expected_contribution','')}",
                    "",
                    f"**Assumptions & risks.** {r.get('assumptions_and_risks','')}",
                    "",
                    "---", "",
                ]
            return "\n".join(out)

        c2.download_button(
            "Download ideas.md", to_md(view).encode("utf-8"),
            file_name="ideas.md", mime="text/markdown",
        )


# =====================================================================
# Tab: Evaluate
# =====================================================================
with tab_eval:
    st.markdown(
        "Run an LLM-as-judge over `artifacts/ideas_full.jsonl` to score each "
        "idea on a 1-5 rubric for **novelty / specificity / feasibility / "
        "evidence_grounding**. Use the CLI for a full batch:"
    )
    st.code("python -m gap2idea.cli evaluate-ideas --judge-model gpt-4.1-mini", language="bash")

    if idea_eval.empty:
        st.info("No evaluation results yet. Generate ideas, then run the command above.")
    else:
        st.markdown(f"**{len(idea_eval)} evaluated ideas.**")
        axes = ["novelty", "specificity", "feasibility", "evidence_grounding", "composite"]
        means = {a: pd.to_numeric(idea_eval[a], errors="coerce").mean() for a in axes if a in idea_eval.columns}
        cols = st.columns(len(means))
        for (k, v), col in zip(means.items(), cols):
            col.metric(k, f"{v:.2f}" if pd.notna(v) else "—")

        st.subheader("Distribution")
        long_df = idea_eval[[a for a in axes if a in idea_eval.columns]].melt(
            var_name="axis", value_name="score",
        )
        box = px.box(long_df, x="axis", y="score", points="all")
        st.plotly_chart(box, use_container_width=True)

        st.subheader("Per-idea scores")
        st.dataframe(
            idea_eval.sort_values("composite", ascending=False),
            use_container_width=True, height=400,
        )

        st.download_button(
            "Download idea_eval.tsv",
            idea_eval.to_csv(sep="\t", index=False).encode("utf-8"),
            file_name="idea_eval.tsv", mime="text/tab-separated-values",
        )
