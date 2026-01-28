#!/usr/bin/env python3
"""
Build FAISS search index for paper titles.

Usage:
    python scripts/build_search_index.py --input data/gaps_with_clusters.tsv --output artifacts/paper_search
    python scripts/build_search_index.py --input data/arxiv-metadata-oai-snapshot.json --output artifacts/paper_search
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from gap2idea.pipeline.semantic_search import (
    SemanticSearch,
    load_papers_from_tsv,
    load_arxiv_jsonl
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index(input_path: str, output_prefix: str, 
                model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                device: str = "cpu", limit: int = None):
    """
    Build FAISS search index from input data.
    
    Args:
        input_path: Path to input TSV or JSONL file
        output_prefix: Prefix for output files (index and metadata)
        model_name: Sentence transformer model name
        device: Device to use for encoding (cpu or cuda)
        limit: Optional limit on number of papers
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load papers based on file type
    logger.info(f"Loading papers from {input_path}...")
    
    if input_path.suffix == ".tsv":
        papers = load_papers_from_tsv(str(input_path))
    elif input_path.suffix in (".json", ".jsonl"):
        papers = load_arxiv_jsonl(str(input_path), limit=limit)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    logger.info(f"Loaded {len(papers)} papers")
    
    if limit and len(papers) > limit:
        papers = papers[:limit]
        logger.info(f"Limited to {limit} papers")
    
    # Build search index
    logger.info(f"Building search index with model {model_name}...")
    engine = SemanticSearch(model_name=model_name, device=device)
    engine.fit(papers)
    
    # Save index
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = f"{output_prefix}.index"
    metadata_path = f"{output_prefix}_metadata.json"
    
    logger.info(f"Saving index to {index_path}...")
    engine.save_index(index_path, metadata_path)
    
    logger.info(f"Saving metadata to {metadata_path}...")
    logger.info(f"Done! Built index with {len(papers)} papers.")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS search index for papers")
    parser.add_argument("--input", required=True, help="Input TSV or JSONL file path")
    parser.add_argument("--output", required=True, help="Output prefix for index files")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for encoding")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of papers")
    
    args = parser.parse_args()
    
    build_index(
        input_path=args.input,
        output_prefix=args.output,
        model_name=args.model,
        device=args.device,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
