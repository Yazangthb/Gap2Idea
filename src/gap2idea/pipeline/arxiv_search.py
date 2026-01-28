"""
ArXiv API search module.

Provides functionality to search arXiv for papers and convert results
to Paper dataclass for use with SemanticSearch.
"""

import re
from typing import List, Optional, Dict, Any

import feedparser

# Note: Paper is imported lazily inside search_arxiv_to_papers to avoid circular import


def _get_arxiv_field(entry: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    """Get arXiv-specific field from feedparser entry."""
    return getattr(entry, f"arxiv_{key}", default)


def search_arxiv(
    query: str,
    max_results: int = 10,
    categories: Optional[List[str]] = None,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    published_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Search arXiv API for papers matching the query.

    Args:
        query: Search query string (e.g., "transformer attention")
        max_results: Maximum number of results to return (default: 10)
        categories: Optional list of arXiv categories to filter (e.g., ['cs.AI', 'cs.LG'])
        sort_by: Sort by 'submittedDate' or 'relevance' (default: 'submittedDate')
        sort_order: Sort order 'descending' or 'ascending' (default: 'descending')
        published_only: If True, only return published papers (default: False)

    Returns:
        List of paper dictionaries with keys:
        - title: Paper title
        - paper_id: arXiv ID
        - abstract: Paper abstract
        - authors: List of author names
        - published: Publication date
        - updated: Last update date
        - arxiv_url: Link to arXiv page
        - pdf_url: Link to PDF
        - doi: DOI if available
        - journal_ref: Journal reference if available
        - published_signal: Boolean indicating if published
    """
    base_url = "http://export.arxiv.org/api/query?"

    # Build search terms
    search_terms = f"all:{query}"
    if categories:
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        search_terms = f"({search_terms}) AND ({cat_query})"

    # Build URL
    url = (
        f"{base_url}"
        f"search_query={search_terms.replace(' ', '+')}"
        f"&sortBy={sort_by}"
        f"&sortOrder={sort_order}"
        f"&max_results={max_results * 2}"  # Request more to filter published_only
    )

    feed = feedparser.parse(url)

    papers = []
    for entry in feed.entries:
        doi = _get_arxiv_field(entry, "doi", None)
        journal_ref = _get_arxiv_field(entry, "journal_ref", None)

        # Get PDF URL
        pdf_url = None
        for link in entry.links:
            if getattr(link, "type", "") == "application/pdf":
                pdf_url = link.href
                break

        # Determine published signal
        published_signal = bool(journal_ref) or bool(doi)

        # Filter if requested
        if published_only and not published_signal:
            continue

        # Extract arXiv ID from entry.id (e.g., "http://arxiv.org/abs/2501.12345v1")
        arxiv_id_raw = entry.id
        if "/abs/" in arxiv_id_raw:
            arxiv_id = arxiv_id_raw.split("/abs/")[1].split("v")[0]
        else:
            arxiv_id = arxiv_id_raw

        papers.append({
            "paper_id": arxiv_id,
            "title": entry.title.strip(),
            "abstract": re.sub(r"\s+", " ", entry.summary).strip(),
            "authors": [a.name for a in entry.authors],
            "published": entry.published,
            "updated": entry.updated,
            "arxiv_url": entry.id,
            "pdf_url": pdf_url,
            "doi": doi,
            "journal_ref": journal_ref,
            "published_signal": published_signal,
        })

        if len(papers) >= max_results:
            break

    # Sort: published first, then by updated date (newest first)
    papers.sort(
        key=lambda p: (
            not p["published_signal"],  # False (published) comes before True
            p["updated"]
        ),
        reverse=False
    )

    return papers


def search_arxiv_to_papers(
    query: str,
    max_results: int = 10,
    categories: Optional[List[str]] = None,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    published_only: bool = False,
) -> List['Paper']:
    """
    Search arXiv and return results as Paper dataclass objects.

    Args:
        query: Search query string
        max_results: Maximum number of results (default: 10)
        categories: Optional arXiv categories to filter
        sort_by: Sort by 'submittedDate' or 'relevance'
        sort_order: Sort order
        published_only: Only return published papers

    Returns:
        List of Paper objects
    """
    # Lazy import to avoid circular import
    from .semantic_search import Paper
    
    results = search_arxiv(
        query=query,
        max_results=max_results,
        categories=categories,
        sort_by=sort_by,
        sort_order=sort_order,
        published_only=published_only,
    )

    papers = []
    for r in results:
        papers.append(Paper(
            paper_id=r["paper_id"],
            title=r["title"],
            abstract=r["abstract"],
        ))

    return papers


# Example usage
if __name__ == "__main__":
    # Test the search
    results = search_arxiv("transformer attention", max_results=5)
    print(f"Found {len(results)} papers:")
    for r in results:
        print(f"  - {r['title'][:60]}... [published: {r['published_signal']}]")

    # Also test conversion to Paper objects
    papers = search_arxiv_to_papers("transformer attention", max_results=5)
    print(f"\nConverted to {len(papers)} Paper objects:")
    for p in papers:
        print(f"  - {p.paper_id}: {p.title[:60]}...")
