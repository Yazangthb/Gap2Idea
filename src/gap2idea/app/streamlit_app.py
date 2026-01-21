import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
from pathlib import Path
from gap2idea.pipeline.openai_ideas import generate_idea_for_pair
import fitz  # pymupdf
from difflib import SequenceMatcher

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

@st.cache_data
def load_metadata():
    metadata = {}
    with open("data/arxiv-metadata-oai-snapshot.json", "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                metadata[item["id"]] = {"title": item.get("title", ""), "abstract": item.get("abstract", "")}
    return metadata

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

# Get paper titles and clusters
paper_info = []
unique_papers = gaps.drop_duplicates(subset="id")
for _, paper in unique_papers.iterrows():
    pdf_path = Path("data/pdfs") / f"{paper['id']}.pdf"
    title = str(get_pdf_title(pdf_path) if pdf_path.exists() else paper['id'])
    cluster = paper['cluster_id']
    paper_info.append({"id": paper['id'], "title": title, "cluster": cluster})

relevant_papers = []
if query:
    # Compute similarity
    similarities = []
    for p in paper_info:
        sim = SequenceMatcher(None, query.lower(), p['title'].lower()).ratio()
        similarities.append((p, sim))
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    relevant_papers = similarities[:5]  # top 5

if relevant_papers:
    st.subheader("Suggested Papers")
    options = [f"{p['title']} (Cluster {p['cluster']})" for p, sim in relevant_papers]
    selected_option = st.selectbox("Select a paper", options, key="paper_select")
    if selected_option:
        selected_index = options.index(selected_option)
        selected_paper = relevant_papers[selected_index][0]
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

