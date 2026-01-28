import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

import fitz  # pymupdf

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    from gap2idea.pipeline.openai_ideas import generate_idea_for_pair
except ModuleNotFoundError as exc:
    project_root = Path(__file__).resolve().parents[2]
    logger.exception(
        "Failed to import gap2idea; cwd=%s __file__=%s sys.path=%s project_root_contents=%s",
        Path.cwd(),
        Path(__file__).resolve(),
        sys.path,
        [p.name for p in project_root.iterdir()],
    )
    st.error(
        "Failed to import the internal package 'gap2idea'. Check Docker PYTHONPATH or installation."
    )
    raise

try:
    from gap2idea.pipeline.semantic_search import SemanticSearch
except ImportError:
    SemanticSearch = None
    logger.warning("SemanticSearch not available; falling back to string matching")

ART_DIR = "artifacts"

@st.cache_data
def load_tsv(path):
    return pd.read_csv(path, sep="\t")

@st.cache_resource
def load_embeddings(path):
    return np.load(path)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_search_engine():
    """Load or build the semantic search engine."""
    import time
    start = time.time()
    if SemanticSearch is None:
        return None
    
    index_path = f"{ART_DIR}/paper_search.index"
    metadata_path = f"{ART_DIR}/paper_search_metadata.json"
    
    # Try to load existing index
    if Path(index_path).exists() and Path(metadata_path).exists():
        try:
            engine = SemanticSearch.load_index(index_path, metadata_path)
            logger.info(f"Loaded search index in {time.time() - start:.2f}s")
            return engine
        except Exception as e:
            logger.warning(f"Failed to load search index: {e}")
    
    # Build index from arxiv metadata if available
    arxiv_metadata = "data/arxiv-metadata-oai-snapshot.json"
    if Path(arxiv_metadata).exists():
        try:
            from gap2idea.pipeline.semantic_search import load_arxiv_jsonl
            papers = load_arxiv_jsonl(arxiv_metadata, limit=50000)
            engine = SemanticSearch(use_precomputed_abstracts=False)
            logger.info(f"Building search index with {len(papers)} papers...")
            engine.fit(papers)
            engine.save_index(index_path, metadata_path)
            logger.info(f"Built and saved search index in {time.time() - start:.2f}s")
            return engine
        except Exception as e:
            logger.warning(f"Failed to build search index: {e}")
    
    return None

@st.cache_data
def compute_2d_embeddings(X):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
    return tsne.fit_transform(X)

@st.cache_data
def get_pdf_title(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        title = doc.metadata.get('title', '')
        doc.close()
        if title:
            return str(title)
        else:
            # Extract from first page text
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)
            text = page.get_text()
            doc.close()
            lines = text.split('\n')[:10]  # first 10 lines
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.isupper():  # heuristic
                    return str(line)
            return str(pdf_path.stem)
    except:
        return str(pdf_path.stem)

st.set_page_config(page_title="Gap2Idea", layout="wide")
st.title("Gap2Idea — Gap Themes → Evidence → Ideas")

gaps = load_tsv(f"{ART_DIR}/gaps_with_clusters.tsv")
clusters = load_tsv(f"{ART_DIR}/cluster_summary.tsv")
pairs = load_tsv(f"{ART_DIR}/cluster_pairs.tsv")
ideas_path = f"{ART_DIR}/ideas_openai.tsv"

ideas = None
try:
    ideas = load_tsv(ideas_path)
except Exception:
    ideas = pd.DataFrame()

embeddings = None
try:
    embeddings = load_embeddings(f"{ART_DIR}/gap_embeddings.npy")
    embedder = load_embedder()
    X_2d = compute_2d_embeddings(embeddings)
except Exception:
    embeddings = None
    X_2d = None
    embedder = None


# Query input
query = st.text_input("Enter a paper title to search", "")

relevant_papers = []
if query:
    # Get paper titles and clusters for fallback
    paper_info = []
    unique_papers = gaps.drop_duplicates(subset="id")
    for _, paper in unique_papers.iterrows():
        pdf_path = Path("data/pdfs") / f"{paper['id']}.pdf"
        title = str(get_pdf_title(pdf_path) if pdf_path.exists() else paper['id'])
        cluster = paper['cluster_id']
        paper_info.append({"id": paper['id'], "title": title, "cluster": cluster})
    
    # Try semantic search first
    search_engine = load_search_engine()
    if search_engine is not None:
        try:
            # Use session state to cache results per query
            cache_key = f"search_{query}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = search_engine.search(query, top_k=10, stage1_candidates=100, w_title=0.4, w_abs=0.6)
            hits = st.session_state[cache_key]
            # Map hits to paper_info format
            for paper, score in hits:
                # Find the paper info in gaps
                paper_rows = gaps[gaps['id'] == paper.paper_id]
                if not paper_rows.empty:
                    cluster_id = int(paper_rows.iloc[0]['cluster_id'])
                    relevant_papers.append({
                        'id': paper.paper_id,
                        'title': paper.title,
                        'cluster': cluster_id,
                        'score': score
                    })
            relevant_papers = relevant_papers[:5]
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
    
    # Fallback to string matching if semantic search didn't return results
    if not relevant_papers:
        from difflib import SequenceMatcher
        similarities = []
        for p in paper_info:
            sim = SequenceMatcher(None, query.lower(), p['title'].lower()).ratio()
            similarities.append((p, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_papers = [{**p, 'score': sim} for p, sim in similarities[:5]]

if relevant_papers:
    st.subheader("Suggested Papers")
    options = [f"{p['title']} (Cluster {p['cluster']}) [{p['score']:.2f}]" for p in relevant_papers]
    selected_option = st.selectbox("Select a paper", options, key="paper_select")
    if selected_option:
        selected_index = options.index(selected_option)
        selected_paper = relevant_papers[selected_index]
        st.session_state['selected_cluster'] = int(selected_paper['cluster'])
        st.write(f"Selected cluster: {selected_paper['cluster']}")

# Plot clusters
if X_2d is not None:
    st.subheader("Cluster Visualization")
    plot_df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'cluster': gaps['cluster_id'],
        'sentence': gaps['gap_sentence']
    })
    fig = px.scatter(plot_df, x='x', y='y', color='cluster', hover_data=['sentence'], custom_data=['cluster'], title="Gap Clusters (t-SNE)")
    selected_points = st.plotly_chart(fig, width='stretch', on_select="rerun", selection_mode="points")
    if selected_points and 'selection' in selected_points and selected_points['selection']['points']:
        selected_cluster = selected_points['selection']['points'][0]['customdata'][0]
        st.session_state['selected_cluster'] = int(selected_cluster)
else:
    st.info("Embeddings not found. Run the pipeline to generate gap_embeddings.npy")

# Cluster selection
st.subheader("Explore Cluster")
cluster_options = clusters["cluster_id"].tolist()
if 'selected_cluster' in st.session_state and st.session_state['selected_cluster'] in cluster_options:
    default_index = cluster_options.index(st.session_state['selected_cluster'])
else:
    default_index = 0
selected_cluster = st.selectbox("Select a cluster to explore", cluster_options, key=f"explore_cluster_{default_index}", index=default_index)

if selected_cluster:
    cluster_papers = gaps[gaps["cluster_id"] == selected_cluster].drop_duplicates(subset="id")
    st.write(f"Papers in cluster {selected_cluster}: {len(cluster_papers)}")
    for _, paper in cluster_papers.iterrows():
        pdf_path = Path("data/pdfs") / f"{paper['id']}.pdf"
        title = str(get_pdf_title(pdf_path) if pdf_path.exists() else paper['id'])
        with st.expander(title):
            st.write(f"**Gap:** {paper['gap_sentence']}")
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label=f"Download PDF",
                    data=pdf_bytes,
                    file_name=f"{paper['id']}.pdf",
                    mime="application/pdf",
                    key=f"pdf_{paper['id']}"
                )
            else:
                st.write("PDF not available")

if selected_cluster:
    st.subheader("Generate Idea")
    if st.button("Generate Idea for this Cluster"):
        # Sample second cluster weighted by similarity
        cluster_pairs = pairs[(pairs['cluster_a'] == selected_cluster) | (pairs['cluster_b'] == selected_cluster)]
        if not cluster_pairs.empty:
            # Get similarities
            sims = []
            for _, row in cluster_pairs.iterrows():
                other = row['cluster_b'] if row['cluster_a'] == selected_cluster else row['cluster_a']
                sim = row['cosine_sim']
                sims.append((other, sim))
            # Weighted sampling
            import random
            others, weights = zip(*sims)
            sampled_other = random.choices(others, weights=weights, k=1)[0]
            
            label_a = clusters[clusters["cluster_id"] == selected_cluster]["theme_label"].iloc[0]
            label_b = clusters[clusters["cluster_id"] == sampled_other]["theme_label"].iloc[0]
            with st.spinner("Generating idea..."):
                idea = generate_idea_for_pair(gaps, selected_cluster, sampled_other, label_a, label_b)
            if idea:
                st.success("Idea generated!")
                st.write(f"**Title:** {idea['title']}")
                st.write(f"**Research Question:** {idea['research_question']}")
                st.write(f"**Method Sketch:** {idea['method_sketch']}")
                st.write(f"**Evaluation Plan:** {idea['evaluation_plan']}")
                st.write(f"**Expected Contribution:** {idea['expected_contribution']}")
                st.write(f"**Assumptions and Risks:** {idea['assumptions_and_risks']}")
                st.write(f"**Confidence:** {idea['idea_confidence']}")
                
                # Why this idea
                st.subheader("Why this Idea?")
                st.write(f"This idea combines gaps from Cluster {selected_cluster} and Cluster {sampled_other}.")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Gaps from Cluster {selected_cluster}:**")
                    ev_a = gaps[gaps["cluster_id"] == selected_cluster].sort_values("confidence", ascending=False).head(3)
                    for _, e in ev_a.iterrows():
                        st.write(f"- {e['gap_sentence']}")
                with col2:
                    st.write(f"**Gaps from Cluster {sampled_other}:**")
                    ev_b = gaps[gaps["cluster_id"] == sampled_other].sort_values("confidence", ascending=False).head(3)
                    for _, e in ev_b.iterrows():
                        st.write(f"- {e['gap_sentence']}")
                
                # Save to ideas
                new_idea = pd.DataFrame([idea])
                if ideas.empty:
                    ideas = new_idea
                else:
                    ideas = pd.concat([ideas, new_idea], ignore_index=True)
                ideas.to_csv(ideas_path, sep="\t", index=False)
                st.info("Idea saved to ideas_openai.tsv")
            else:
                st.error("Failed to generate idea.")
        else:
            st.write("No similar clusters found.")

st.divider()

st.subheader("Missing Connections (Cluster Pairs)")
st.dataframe(pairs.head(30), width='stretch')

if len(pairs) > 0:
    st.subheader("Pair Inspector")
    pidx = st.slider("Pair index", 0, max(0, len(pairs)-1), 0)
    pr = pairs.iloc[pidx]
    ca, cb = int(pr["cluster_a"]), int(pr["cluster_b"])
    st.write(f"**A:** {pr.get('label_a','')} (cluster {ca})")
    st.write(f"**B:** {pr.get('label_b','')} (cluster {cb})")
    st.write(f"Similarity: `{pr['cosine_sim']:.3f}`")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Evidence A")
        ev_a = gaps[gaps.cluster_id==ca].sort_values("confidence", ascending=False).head(10)
        for _, e in ev_a.iterrows():
            st.write(f"- {e['gap_sentence']}")
    with col2:
        st.write("Evidence B")
        ev_b = gaps[gaps.cluster_id==cb].sort_values("confidence", ascending=False).head(10)
        for _, e in ev_b.iterrows():
            st.write(f"- {e['gap_sentence']}")

st.divider()

st.subheader("Generated Ideas (if available)")
if ideas is None or ideas.empty:
    st.info("No ideas_openai.tsv found yet. Run idea generation step to populate.")
else:
    st.dataframe(ideas, width='stretch')

