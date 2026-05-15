"""Gap2Idea MCP server.

Exposes the corpus + idea operations as MCP tools so Claude Desktop /
Cursor / any MCP-aware client can query your literature directly.

Configure Claude Desktop by adding to
  %APPDATA%\\Claude\\claude_desktop_config.json (Windows)
  ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
  ~/.config/Claude/claude_desktop_config.json (Linux)

  {
    "mcpServers": {
      "gap2idea": {
        "command": "gap2idea",
        "args": ["serve-mcp", "--root", "C:/path/to/your/Gap2Idea"]
      }
    }
  }

Then ask Claude things like:
  "Use the gap2idea tool to list my themes, then show me ideas in the
   library with novelty >= 0.7 and confidence >= 0.85."
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from gap2idea import tools as T
from gap2idea.utils import get_logger

log = get_logger("gap2idea.mcp")


def build_server(root: str | Path = ".") -> FastMCP:
    """Construct a FastMCP server pre-bound to a corpus root.

    Each tool is registered as a thin wrapper that calls `tools.dispatch`
    with the right `root`. Keeping the bindings explicit (rather than a
    generic dispatch tool) gives Claude a clearer schema to reason over.
    """
    root_str = str(Path(root).resolve())
    mcp = FastMCP(
        "gap2idea",
        instructions=(
            "Tools for browsing a Gap2Idea research-gap corpus and synthesising / "
            "fact-checking research ideas. The corpus root is fixed at server "
            f"start ({root_str}). Read tools (list_themes, get_theme, get_evidence, "
            "list_ideas, get_idea) are safe to call freely. Write tools (save_idea) "
            "modify artifacts/ideas.tsv on disk."
        ),
    )

    # ---- read ----

    @mcp.tool()
    async def list_themes() -> list[dict]:
        """List every research-gap theme (cluster) with size and label."""
        return await T.list_themes(root=root_str)

    @mcp.tool()
    async def get_theme(cluster_id: int) -> dict:
        """Full details of one theme: label, keywords, papers, gap sentences."""
        return await T.get_theme(cluster_id, root=root_str)

    @mcp.tool()
    async def get_evidence(cluster_id: int, k: int = 4) -> list[dict]:
        """Sample k diverse evidence sentences from a theme."""
        return await T.get_evidence(cluster_id, k=k, root=root_str)

    @mcp.tool()
    async def retrieve_methods(
        cluster_id: int,
        top_k: int = 5,
        sim_low: float = 0.30,
        sim_high: float = 0.70,
    ) -> list[dict]:
        """Retrieve method-claim sentences in the sim sweet spot for a theme.

        Used to build "apply method M to gap G" ideas (method-gap mode).
        """
        return await T.retrieve_methods(
            cluster_id, top_k=top_k, sim_low=sim_low, sim_high=sim_high, root=root_str,
        )

    @mcp.tool()
    async def search_prior_art(query: str, limit: int = 10) -> list[dict]:
        """Search Semantic Scholar for papers matching a query."""
        return await T.search_prior_art(query, limit=limit)

    @mcp.tool()
    async def score_novelty(
        title: str, research_question: str, method_sketch: str = "", top_k: int = 10,
    ) -> dict:
        """Estimate idea novelty: 1 - max_cosine(idea, S2_hit_abstract)."""
        return await T.score_novelty(title, research_question, method_sketch, top_k)

    @mcp.tool()
    async def check_evidence_overlap(
        evidence_used: list[dict], fed_evidence: list[dict],
    ) -> dict:
        """Fraction of evidence_used that was actually in fed_evidence.

        evidence_used: [{paper_id, gap_sentence}, ...] as the LLM claimed.
        fed_evidence:  [{paper_id, gap_sentence}, ...] you actually fed it.
        Returns {overlap, n_used, n_grounded, hallucinated}.
        """
        return await T.check_evidence_overlap(evidence_used, fed_evidence)

    @mcp.tool()
    async def list_ideas(
        min_novelty: float = 0.0,
        min_confidence: float = 0.0,
        mode: str = "any",
    ) -> list[dict]:
        """List saved ideas. Filter by mode (bridge|within|method-gap|any), min_novelty, min_confidence."""
        return await T.list_ideas(
            min_novelty=min_novelty, min_confidence=min_confidence, mode=mode, root=root_str,
        )

    @mcp.tool()
    async def get_idea(idx: int) -> dict:
        """Get one saved idea by row index (as listed by list_ideas)."""
        return await T.get_idea(idx, root=root_str)

    @mcp.tool()
    async def save_idea(idea: dict) -> dict:
        """Append an idea (flat dict) to artifacts/ideas.tsv. Returns {saved, total_ideas}."""
        return await T.save_idea(idea, root=root_str)

    return mcp


def run_stdio(root: str | Path = ".") -> None:
    """Block on stdin/stdout MCP protocol. Used by `gap2idea serve-mcp`."""
    server = build_server(root)
    server.run("stdio")


if __name__ == "__main__":
    run_stdio(os.environ.get("GAP2IDEA_ROOT", "."))
