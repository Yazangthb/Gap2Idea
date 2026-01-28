"""
Semantic search module using FAISS and sentence-transformers.

This module provides hierarchical semantic search:
- Stage 1: Title-only FAISS retrieval
- Stage 2: Rerank topN using abstract

Score: score = w_title*cos(q, title) + w_abs*cos(q, abstract)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Paper:
    """Represents a paper with ID, title, and abstract."""
    paper_id: str
    title: str
    abstract: str


def load_arxiv_jsonl(jsonl_path: str, limit: Optional[int] = None) -> List[Paper]:
    """Load papers from arxiv JSONL file."""
    papers = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = _parse_json_line(line)
            pid = obj.get("id", "")
            title = _clean_text(obj.get("title", ""))
            abstract = _clean_text(obj.get("abstract", ""))
            if pid and title:
                papers.append(Paper(pid, title, abstract))
            if limit is not None and len(papers) >= limit:
                break
    return papers


def load_papers_from_tsv(tsv_path: str, id_col: str = "id", 
                         title_col: str = "title", 
                         abstract_col: str = "abstract") -> List[Paper]:
    """Load papers from TSV file."""
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t")
    papers = []
    for _, row in df.iterrows():
        paper_id = str(row.get(id_col, ""))
        title = _clean_text(str(row.get(title_col, "")))
        abstract = _clean_text(str(row.get(abstract_col, "")))
        if paper_id and title:
            papers.append(Paper(paper_id, title, abstract))
    return papers


def _parse_json_line(line: str) -> Dict:
    """Safely parse a JSON line."""
    try:
        import json
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def _clean_text(text: str) -> str:
    """Clean and normalize text."""
    if text is None:
        return ""
    return text.strip().replace("\n", " ")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize vectors along axis 1."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def build_ip_index(vectors: np.ndarray) -> faiss.Index:
    """Build FAISS Inner Product index from normalized vectors."""
    d = vectors.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(vectors.astype(np.float32))
    return idx


class SemanticSearch:
    """
    Hierarchical semantic search using FAISS.
    
    Stage 1: Title-only FAISS retrieval
    Stage 2: Rerank topN using abstract (precomputed or on-the-fly)
    
    Score: score = w_title*cos(q, title) + w_abs*cos(q, abstract)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 128,
        use_precomputed_abstracts: bool = True,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.use_precomputed_abstracts = use_precomputed_abstracts
        
        self.papers: List[Paper] = []
        self.title_vecs: Optional[np.ndarray] = None
        self.abs_vecs: Optional[np.ndarray] = None
        
        self.title_index: Optional[faiss.Index] = None
        self.id_to_idx: Dict[str, int] = {}
    
    def fit(self, papers: List[Paper]) -> None:
        """
        Build search index from list of papers.
        
        Args:
            papers: List of Paper objects
        """
        self.papers = papers
        self.id_to_idx = {p.paper_id: i for i, p in enumerate(papers)}
        
        # Generate title embeddings
        titles = [p.title for p in papers]
        title_vecs = self.model.encode(
            titles, 
            batch_size=self.batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        title_vecs = l2_normalize(title_vecs).astype(np.float32)
        self.title_vecs = title_vecs
        self.title_index = build_ip_index(self.title_vecs)
        
        # Optionally generate abstract embeddings
        if self.use_precomputed_abstracts:
            abstracts = [p.abstract if p.abstract else "" for p in papers]
            abs_vecs = self.model.encode(
                abstracts, 
                batch_size=self.batch_size, 
                show_progress_bar=True, 
                convert_to_numpy=True
            )
            abs_vecs = l2_normalize(abs_vecs).astype(np.float32)
            self.abs_vecs = abs_vecs
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """Save FAISS index and paper metadata to disk."""
        if self.title_index is None:
            raise RuntimeError("Call fit() first.")
        
        # Save FAISS index
        faiss.write_index(self.title_index, index_path)
        
        # Save metadata (paper info)
        import json
        metadata = []
        for p in self.papers:
            metadata.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract
            })
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_index(cls, index_path: str, metadata_path: str,
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                   device: str = "cpu") -> "SemanticSearch":
        """Load saved FAISS index and metadata."""
        import json
        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        papers = [Paper(p["paper_id"], p["title"], p["abstract"]) for p in metadata]
        
        # Create instance and load index
        instance = cls(model_name=model_name, device=device)
        instance.papers = papers
        instance.id_to_idx = {p.paper_id: i for i, p in enumerate(papers)}
        instance.title_index = faiss.read_index(index_path)
        instance.title_vecs = None  # Not needed for search
        instance.abs_vecs = None
        
        return instance
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        stage1_candidates: int = 100,
        w_title: float = 0.4,
        w_abs: float = 0.6,
    ) -> List[Tuple[Paper, float]]:
        """
        Search for papers matching the query.
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            stage1_candidates: Number of candidates from stage 1 (title search)
            w_title: Weight for title similarity
            w_abs: Weight for abstract similarity
            
        Returns:
            List of (Paper, score) tuples sorted by score descending
        """
        if self.title_index is None:
            raise RuntimeError("Call fit() or load_index() first.")
        
        # Embed query once
        q = self.model.encode([query], convert_to_numpy=True)
        q = l2_normalize(q).astype(np.float32)
        
        # Stage 1: title retrieval using FAISS
        scores1, ids1 = self.title_index.search(q, stage1_candidates)
        cand_ids = [i for i in ids1[0].tolist() if i != -1]
        
        # Stage 2: rerank using weighted title+abstract similarity
        results = []
        for idx in cand_ids:
            s_title = float(np.dot(q[0], self.title_vecs[idx]))
            
            # Abstract similarity
            if self.abs_vecs is not None:
                s_abs = float(np.dot(q[0], self.abs_vecs[idx]))
            else:
                # Compute abstract embedding on-the-fly
                abs_text = self.papers[idx].abstract or ""
                abs_vec = self.model.encode([abs_text], convert_to_numpy=True)
                abs_vec = l2_normalize(abs_vec).astype(np.float32)
                s_abs = float(np.dot(q[0], abs_vec[0]))
            
            score = w_title * s_title + w_abs * s_abs
            results.append((idx, score))
        
        # Sort and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        return [(self.papers[i], s) for i, s in results]


# Example usage
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Example: Load from arxiv metadata
    arxiv_path = os.path.expanduser("~/.cache/kagglehub/datasets/Cornell-University/arxiv/*/arxiv-metadata-oai-snapshot.json")
    arxiv_files = Path(".").glob(arxiv_path)
    arxiv_file = str(list(arxiv_files)[0]) if list(arxiv_files) else None
    
    if arxiv_file and Path(arxiv_file).exists():
        papers = load_arxiv_jsonl(arxiv_file, limit=10000)
        print(f"Loaded {len(papers)} papers")
        
        # Build search index
        engine = SemanticSearch(device="cpu")
        engine.fit(papers)
        
        # Save index
        engine.save_index("artifacts/paper_title.index", "artifacts/paper_metadata.json")
        
        # Search example
        query = "retrieval augmented generation for scientific literature"
        hits = engine.search(query, top_k=10, stage1_candidates=100, w_title=0.4, w_abs=0.6)
        
        for p, s in hits:
            print(f"{s:.4f} | {p.paper_id} | {p.title[:120]}")
