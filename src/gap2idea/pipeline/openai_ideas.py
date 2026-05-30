"""Generate research ideas from gap clusters in one of three modes.

  - `bridge`     : pair two gap-clusters that sit in the bridge-score sweet
                   spot, ask the LLM to combine them. Default. Encodes
                   "novelty by recombining related-but-distinct themes."
  - `within`     : for each gap-cluster, ask the LLM to synthesise ONE idea
                   that addresses the recurring opportunity that cluster
                   represents. No pairing — uses all evidence within the
                   cluster.
  - `method-gap` : for each gap-cluster, retrieve method-claim sentences
                   (from `data/methods.tsv`) whose similarity to the cluster
                   centroid is in the sweet spot, then ask the LLM to apply
                   those methods to those gaps. Explicit "X solves Y."

All three modes share:
1. Diverse evidence picking (greedy Jaccard maximisation).
2. Strict JSON-schema LLM call with the same idea schema.
3. Semantic Scholar novelty check on the generated title.
4. TSV (flat) + JSONL (full provenance) outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from gap2idea.pipeline.llm import DEFAULT_MODEL, get_llm_client, parse_json_response
from gap2idea.pipeline.semantic_scholar import S2Client
from gap2idea.utils import get_logger, retry

log = get_logger(__name__)

IDEA_MODEL = DEFAULT_MODEL
DEFAULT_K_PER_CLUSTER = 4


IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "pair": {
            "type": "object",
            "properties": {
                "cluster_a": {"type": "integer"},
                "cluster_b": {"type": "integer"},
            },
            "required": ["cluster_a", "cluster_b"],
            "additionalProperties": False,
        },
        "idea": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "research_question": {"type": "string"},
                "method_sketch": {"type": "string"},
                "evaluation_plan": {"type": "string"},
                "expected_contribution": {"type": "string"},
                "assumptions_and_risks": {"type": "string"},
                "evidence_used": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string"},
                            "gap_sentence": {"type": "string"},
                        },
                        "required": ["paper_id", "gap_sentence"],
                        "additionalProperties": False,
                    },
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "title",
                "research_question",
                "method_sketch",
                "evaluation_plan",
                "expected_contribution",
                "assumptions_and_risks",
                "evidence_used",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["pair", "idea"],
    "additionalProperties": False,
}

SYSTEM_IDEA = (
    "You are a research planning assistant. "
    "Use ONLY the provided evidence sentences and paragraphs. "
    "Do NOT invent datasets, results, or claims. "
    "Propose exactly one actionable, novel research idea that meaningfully "
    "combines both gap themes. The method must be concrete enough that a "
    "graduate student could begin work next week. The evaluation plan must "
    "name a metric and a comparison baseline."
)


# ---------- evidence selection ----------

def _diverse_evidence(
    gaps: pd.DataFrame,
    cluster_id: int,
    k: int = DEFAULT_K_PER_CLUSTER,
) -> pd.DataFrame:
    """Pick `k` evidence rows from `cluster_id`, balancing confidence + diversity.

    Strategy: take the highest-confidence row first, then iteratively pick the
    row whose gap_sentence is *least similar* (by token Jaccard) to anything
    already picked. This is cheap and avoids feeding the LLM near-duplicates.
    """
    sub = gaps[gaps["cluster_id"] == cluster_id].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("confidence", ascending=False).reset_index(drop=True)
    if len(sub) <= k:
        return sub

    def toks(s: str) -> set[str]:
        return {w.lower() for w in str(s).split() if len(w) > 3}

    picked_idx: list[int] = [0]
    picked_toks: list[set[str]] = [toks(sub.iloc[0]["gap_sentence"])]
    candidates = list(range(1, len(sub)))

    while len(picked_idx) < k and candidates:
        best_i, best_score = None, -1.0
        for i in candidates:
            ti = toks(sub.iloc[i]["gap_sentence"])
            # min jaccard to anything already picked: lower = more diverse
            min_j = min(
                len(ti & tj) / max(1, len(ti | tj)) for tj in picked_toks
            )
            score = (1.0 - min_j) * 0.7 + float(sub.iloc[i]["confidence"]) * 0.3
            if score > best_score:
                best_score, best_i = score, i
        assert best_i is not None
        picked_idx.append(best_i)
        picked_toks.append(toks(sub.iloc[best_i]["gap_sentence"]))
        candidates.remove(best_i)

    return sub.iloc[picked_idx].reset_index(drop=True)


def _evidence_payload(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "paper_id": str(r["id"]),
                "gap_type": str(r.get("gap_type", "")),
                "section_type": str(r.get("section_type", "")),
                "confidence": float(r.get("confidence", 0.0)),
                "gap_sentence": str(r["gap_sentence"]),
                "paragraph_text": str(r.get("paragraph_text", "")),
            }
        )
    return out


# ---------- prompt ----------

def _build_user_prompt(
    cluster_a: int, cluster_b: int, label_a: str, label_b: str,
    ev_a: list[dict], ev_b: list[dict],
) -> str:
    payload = {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "theme_a_label": label_a,
        "theme_b_label": label_b,
        "evidence_a": ev_a,
        "evidence_b": ev_b,
    }
    return (
        "Return JSON matching the schema.\n"
        "Constraints:\n"
        "- Use ONLY evidence_a/evidence_b content; do not introduce papers, datasets, or numbers not present there.\n"
        "- Method must be testable, with a named evaluation metric and at least one baseline.\n"
        "- evidence_used must be a subset of provided evidence sentences (paper_id + verbatim gap_sentence).\n"
        "- title: a precise, neutral, conference-paper-style title (<=15 words).\n"
        "- assumptions_and_risks: 2-4 sentences naming the most load-bearing assumption AND the most likely way this could fail.\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False)
    )


@retry(tries=3, base_delay=2.0)
def _call_llm(client: OpenAI, prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_IDEA},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "idea_synthesis", "schema": IDEA_SCHEMA, "strict": True},
        },
        temperature=0.7,
    )
    return parse_json_response(resp.choices[0].message.content)


# ---------- novelty check ----------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def novelty_check(
    idea: dict,
    s2: S2Client,
    embedder,
    top_k: int = 10,
) -> dict:
    """Compare the idea against Semantic Scholar to estimate novelty.

    Returns:
        {
          'max_similarity':     max cosine(idea, hit_abstract) over top_k S2 hits,
          'mean_similarity':    mean over those,
          'novelty_score':      1 - max_similarity, clipped to [0, 1],
          'closest_paper': {paperId, title, year, similarity} or None,
          'n_hits':             number of S2 hits considered,
        }
    """
    query = f"{idea['title']}. {idea['research_question']}"
    try:
        hits = s2.search(query, limit=top_k)
    except Exception as e:
        log.warning("S2 search failed during novelty check: %s", e)
        return {
            "max_similarity": None, "mean_similarity": None,
            "novelty_score": None, "closest_paper": None, "n_hits": 0,
        }

    # Embed the idea once.
    idea_text = f"{idea['title']}. {idea['research_question']}. {idea['method_sketch']}"
    idea_vec = embedder.encode([idea_text], normalize_embeddings=True)[0]

    sims: list[tuple[float, dict]] = []
    for h in hits:
        ab = h.get("abstract") or h.get("title") or ""
        if not ab.strip():
            continue
        hv = embedder.encode([ab], normalize_embeddings=True)[0]
        sims.append((_cosine(idea_vec, hv), h))

    if not sims:
        return {
            "max_similarity": 0.0, "mean_similarity": 0.0,
            "novelty_score": 1.0, "closest_paper": None, "n_hits": 0,
        }

    sims.sort(key=lambda x: x[0], reverse=True)
    max_sim, closest = sims[0]
    mean_sim = float(np.mean([s for s, _ in sims]))
    return {
        "max_similarity": float(max_sim),
        "mean_similarity": mean_sim,
        "novelty_score": float(max(0.0, min(1.0, 1.0 - max_sim))),
        "closest_paper": {
            "paperId": closest.get("paperId"),
            "title": closest.get("title"),
            "year": closest.get("year"),
            "similarity": float(max_sim),
        },
        "n_hits": len(sims),
    }


# ---------- one-pair API used by Streamlit ----------

def generate_idea_for_pair(
    gaps: pd.DataFrame,
    cluster_a: int,
    cluster_b: int,
    label_a: str,
    label_b: str,
    model: str = IDEA_MODEL,
    check_novelty: bool = True,
    k_evidence: int = DEFAULT_K_PER_CLUSTER,
) -> dict | None:
    """Generate one idea + novelty check. Returns a flat dict for TSV row."""
    client = get_llm_client()
    ev_a_df = _diverse_evidence(gaps, cluster_a, k=k_evidence)
    ev_b_df = _diverse_evidence(gaps, cluster_b, k=k_evidence)
    if ev_a_df.empty or ev_b_df.empty:
        return None

    ev_a = _evidence_payload(ev_a_df)
    ev_b = _evidence_payload(ev_b_df)
    prompt = _build_user_prompt(cluster_a, cluster_b, label_a, label_b, ev_a, ev_b)
    data = _call_llm(client, prompt, model=model)
    idea = data["idea"]

    nov = {"novelty_score": None, "max_similarity": None, "closest_paper": None}
    if check_novelty:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            nov = novelty_check(idea, S2Client(), embedder, top_k=10)
        except Exception as e:
            log.warning("Novelty check failed: %s", e)

    closest = nov.get("closest_paper") or {}
    return {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "label_a": label_a,
        "label_b": label_b,
        "title": idea["title"],
        "research_question": idea["research_question"],
        "method_sketch": idea["method_sketch"],
        "evaluation_plan": idea["evaluation_plan"],
        "expected_contribution": idea["expected_contribution"],
        "assumptions_and_risks": idea["assumptions_and_risks"],
        "idea_confidence": idea["confidence"],
        "evidence_used_json": json.dumps(idea["evidence_used"], ensure_ascii=False),
        "novelty_score": nov.get("novelty_score"),
        "max_similarity_to_prior": nov.get("max_similarity"),
        "closest_paper_title": closest.get("title", ""),
        "closest_paper_year": closest.get("year", ""),
        "closest_paper_id": closest.get("paperId", ""),
    }


# ---------- batch generation over a pairs file ----------

def generate_ideas_batch(
    gaps: pd.DataFrame,
    pairs: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    out_tsv: Path,
    out_jsonl: Path,
    n_pairs: int | None = None,
    model: str = IDEA_MODEL,
    check_novelty: bool = True,
    k_evidence: int = DEFAULT_K_PER_CLUSTER,
) -> pd.DataFrame:
    """Generate ideas for the top-`n_pairs` rows of `pairs` (sorted by bridge_score).

    Writes:
      - `out_tsv` : flat one-row-per-idea TSV for the Streamlit dashboard.
      - `out_jsonl` : one rich record per idea (prompt + response + novelty
        payload + full evidence) for offline analysis / paper appendix.
    """
    label_map = dict(zip(cluster_labels["cluster_id"].astype(int), cluster_labels["theme_label"]))
    if "bridge_score" in pairs.columns:
        pairs = pairs.sort_values("bridge_score", ascending=False)
    if n_pairs:
        pairs = pairs.head(n_pairs)

    client = get_llm_client()

    embedder = None
    s2 = None
    if check_novelty:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            s2 = S2Client()
        except Exception as e:
            log.warning("Disabling novelty check: %s", e)
            check_novelty = False

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    flat_rows: list[dict] = []
    with open(out_jsonl, "w", encoding="utf-8") as jf:
        for i, row in enumerate(pairs.itertuples(index=False), 1):
            ca, cb = int(row.cluster_a), int(row.cluster_b)
            la = label_map.get(ca, "")
            lb = label_map.get(cb, "")
            ev_a_df = _diverse_evidence(gaps, ca, k=k_evidence)
            ev_b_df = _diverse_evidence(gaps, cb, k=k_evidence)
            if ev_a_df.empty or ev_b_df.empty:
                log.warning("Skipping pair (%d,%d): empty evidence", ca, cb)
                continue
            ev_a = _evidence_payload(ev_a_df)
            ev_b = _evidence_payload(ev_b_df)
            prompt = _build_user_prompt(ca, cb, la, lb, ev_a, ev_b)
            try:
                data = _call_llm(client, prompt, model=model)
            except Exception as e:
                log.error("LLM call failed for pair (%d,%d): %s", ca, cb, e)
                continue
            idea = data["idea"]

            nov = {}
            if check_novelty:
                try:
                    nov = novelty_check(idea, s2, embedder, top_k=10)
                except Exception as e:
                    log.warning("Novelty check failed for pair (%d,%d): %s", ca, cb, e)

            closest = nov.get("closest_paper") or {}
            flat = {
                "mode": "bridge",
                "cluster_a": ca,
                "cluster_b": cb,
                "label_a": la,
                "label_b": lb,
                "bridge_score": float(getattr(row, "bridge_score", 0.0) or 0.0),
                "cosine_sim": float(getattr(row, "cosine_sim", 0.0) or 0.0),
                "title": idea["title"],
                "research_question": idea["research_question"],
                "method_sketch": idea["method_sketch"],
                "evaluation_plan": idea["evaluation_plan"],
                "expected_contribution": idea["expected_contribution"],
                "assumptions_and_risks": idea["assumptions_and_risks"],
                "idea_confidence": float(idea["confidence"]),
                "evidence_used_json": json.dumps(idea["evidence_used"], ensure_ascii=False),
                "novelty_score": nov.get("novelty_score"),
                "max_similarity_to_prior": nov.get("max_similarity"),
                "closest_paper_title": closest.get("title", ""),
                "closest_paper_year": closest.get("year", ""),
                "closest_paper_id": closest.get("paperId", ""),
            }
            flat_rows.append(flat)

            full = {
                "mode": "bridge",
                "pair": {"cluster_a": ca, "cluster_b": cb, "label_a": la, "label_b": lb},
                "evidence_a": ev_a,
                "evidence_b": ev_b,
                "prompt": prompt,
                "system": SYSTEM_IDEA,
                "model": model,
                "idea": idea,
                "novelty": nov,
            }
            jf.write(json.dumps(full, ensure_ascii=False) + "\n")
            log.info("  [%d] (%d,%d) %s  nov=%s", i, ca, cb, idea["title"][:70],
                     f"{nov.get('novelty_score'):.2f}" if nov.get("novelty_score") is not None else "n/a")

    out_df = pd.DataFrame(flat_rows)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote %d ideas to %s and %s", len(out_df), out_tsv, out_jsonl)
    return out_df


# ===========================================================================
# Option A: within-cluster synthesis
# ===========================================================================

SYSTEM_WITHIN = (
    "You are a research planning assistant. "
    "You are given a recurring research gap that appears across multiple papers. "
    "Use ONLY the provided evidence sentences and paragraphs. "
    "Do NOT invent datasets, results, or claims. "
    "Propose exactly one actionable, novel research idea that addresses the "
    "unifying opportunity these gaps represent. The method must be concrete "
    "enough that a graduate student could begin work next week. The evaluation "
    "plan must name a metric and a comparison baseline."
)


def _build_within_prompt(cluster_id: int, label: str, evidence: list[dict]) -> str:
    payload = {
        "cluster_id": cluster_id,
        "theme_label": label,
        "n_evidence": len(evidence),
        "n_papers": len({e["paper_id"] for e in evidence}),
        "evidence": evidence,
    }
    return (
        "Return JSON matching the schema.\n"
        "Constraints:\n"
        "- Use ONLY the `evidence` content; do not introduce papers, datasets, or numbers not present there.\n"
        "- Method must be testable, with a named evaluation metric and at least one baseline.\n"
        "- evidence_used must be a subset of provided evidence sentences (paper_id + verbatim gap_sentence).\n"
        "- title: a precise, conference-paper-style title (<=15 words).\n"
        "- assumptions_and_risks: 2-4 sentences naming the most load-bearing assumption AND the most likely failure mode.\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False)
    )


@retry(tries=3, base_delay=2.0)
def _call_llm_within(client: OpenAI, prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_WITHIN},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "idea_synthesis", "schema": IDEA_SCHEMA, "strict": True},
        },
        temperature=0.7,
    )
    return parse_json_response(resp.choices[0].message.content)


def generate_ideas_within_clusters(
    gaps: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    out_tsv: Path,
    out_jsonl: Path,
    n_clusters: int | None = None,
    model: str = IDEA_MODEL,
    check_novelty: bool = True,
    k_evidence: int = 8,
) -> pd.DataFrame:
    """One idea per cluster. Uses k diverse evidence rows from that cluster only."""
    label_map = dict(zip(cluster_labels["cluster_id"].astype(int), cluster_labels["theme_label"]))
    cids = sorted(c for c in gaps["cluster_id"].unique() if c != -1)
    if n_clusters:
        # Prefer larger clusters (more evidence => better synthesis target)
        sizes = gaps[gaps["cluster_id"].isin(cids)].groupby("cluster_id").size().sort_values(ascending=False)
        cids = sizes.head(n_clusters).index.tolist()

    client = get_llm_client()
    embedder = None
    s2 = None
    if check_novelty:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            s2 = S2Client()
        except Exception as e:
            log.warning("Disabling novelty check: %s", e)
            check_novelty = False

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    flat_rows: list[dict] = []
    with open(out_jsonl, "w", encoding="utf-8") as jf:
        for i, cid in enumerate(cids, 1):
            label = label_map.get(int(cid), "")
            ev_df = _diverse_evidence(gaps, int(cid), k=k_evidence)
            if ev_df.empty:
                log.warning("Skipping cluster %s: empty evidence", cid)
                continue
            evidence = _evidence_payload(ev_df)
            prompt = _build_within_prompt(int(cid), label, evidence)
            try:
                data = _call_llm_within(client, prompt, model=model)
            except Exception as e:
                log.error("LLM call failed for cluster %s: %s", cid, e)
                continue
            idea = data["idea"]

            nov = {}
            if check_novelty:
                try:
                    nov = novelty_check(idea, s2, embedder, top_k=10)
                except Exception as e:
                    log.warning("Novelty check failed for cluster %s: %s", cid, e)

            closest = nov.get("closest_paper") or {}
            flat_rows.append({
                "mode": "within",
                "cluster_a": int(cid),
                "cluster_b": None,
                "label_a": label,
                "label_b": "",
                "bridge_score": None,
                "cosine_sim": None,
                "title": idea["title"],
                "research_question": idea["research_question"],
                "method_sketch": idea["method_sketch"],
                "evaluation_plan": idea["evaluation_plan"],
                "expected_contribution": idea["expected_contribution"],
                "assumptions_and_risks": idea["assumptions_and_risks"],
                "idea_confidence": float(idea["confidence"]),
                "evidence_used_json": json.dumps(idea["evidence_used"], ensure_ascii=False),
                "novelty_score": nov.get("novelty_score"),
                "max_similarity_to_prior": nov.get("max_similarity"),
                "closest_paper_title": closest.get("title", ""),
                "closest_paper_year": closest.get("year", ""),
                "closest_paper_id": closest.get("paperId", ""),
            })

            jf.write(json.dumps({
                "mode": "within",
                "pair": {"cluster_a": int(cid), "label_a": label},
                "evidence_a": evidence,
                "prompt": prompt,
                "system": SYSTEM_WITHIN,
                "model": model,
                "idea": idea,
                "novelty": nov,
            }, ensure_ascii=False) + "\n")
            log.info("  [%d/%d] cluster %d (%s) %s  nov=%s",
                     i, len(cids), int(cid), label[:30], idea["title"][:60],
                     f"{nov.get('novelty_score'):.2f}" if nov.get("novelty_score") is not None else "n/a")

    out_df = pd.DataFrame(flat_rows)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote %d within-cluster ideas to %s and %s", len(out_df), out_tsv, out_jsonl)
    return out_df


# ===========================================================================
# Option B: method x gap matching
# ===========================================================================

SYSTEM_METHOD_GAP = (
    "You are a research planning assistant. "
    "You will receive a CLUSTER OF GAPS (research opportunities) and a "
    "CANDIDATE METHODS list (capabilities other papers have already shown work). "
    "Propose exactly one actionable research idea of the form 'apply method M "
    "(paper P1) to address gap G (paper P2).' "
    "Use ONLY the provided evidence sentences and paragraphs. "
    "Do NOT invent datasets, results, or claims. "
    "The method must be testable; the evaluation plan must name a metric and "
    "comparison baseline."
)


def _build_method_gap_prompt(
    cluster_id: int, label: str, gap_evidence: list[dict], method_evidence: list[dict]
) -> str:
    payload = {
        "gap_cluster_id": cluster_id,
        "gap_cluster_label": label,
        "gap_evidence": gap_evidence,
        "candidate_methods": method_evidence,
    }
    return (
        "Return JSON matching the schema.\n"
        "Constraints:\n"
        "- The idea must explicitly apply ONE of the candidate methods to the gap_cluster.\n"
        "- Use ONLY content from gap_evidence and candidate_methods.\n"
        "- evidence_used must include BOTH (paper_id, gap_sentence) pairs from gap_evidence\n"
        "  AND (paper_id, method_sentence-as-gap_sentence) pairs from candidate_methods\n"
        "  -- i.e. list every supporting input you actually used.\n"
        "- title: precise conference-paper-style title (<=15 words).\n"
        "- assumptions_and_risks: 2-4 sentences naming the load-bearing assumption + failure mode.\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False)
    )


@retry(tries=3, base_delay=2.0)
def _call_llm_method_gap(client: OpenAI, prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_METHOD_GAP},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "idea_synthesis", "schema": IDEA_SCHEMA, "strict": True},
        },
        temperature=0.7,
    )
    return parse_json_response(resp.choices[0].message.content)


def _retrieve_methods_for_cluster(
    cluster_id: int,
    gap_embeddings: np.ndarray,
    gaps: pd.DataFrame,
    methods: pd.DataFrame,
    method_embeddings: np.ndarray,
    top_k: int = 5,
    sim_low: float = 0.30,
    sim_high: float = 0.70,
) -> pd.DataFrame:
    """For a gap-cluster, return the top_k method sentences whose cosine
    similarity to the cluster centroid is in (sim_low, sim_high).

    The "sweet spot" filter avoids:
      - methods that are TOO similar (essentially restating the gap)
      - methods that are TOO distant (unrelated to the gap domain)
    """
    if methods.empty or method_embeddings is None or method_embeddings.size == 0:
        return methods.iloc[0:0]
    idx_in_cluster = np.where(gaps["cluster_id"].values == cluster_id)[0]
    if len(idx_in_cluster) == 0:
        return methods.iloc[0:0]
    centroid = gap_embeddings[idx_in_cluster].mean(axis=0)
    centroid = centroid / max(1e-9, np.linalg.norm(centroid))
    # rows of method_embeddings are already L2-normalised
    sims = method_embeddings @ centroid
    mask = (sims >= sim_low) & (sims <= sim_high)
    if not mask.any():
        # fall back: just take the closest top_k without the sweet-spot filter
        order = np.argsort(-sims)[:top_k]
    else:
        # rank candidates by closeness to the centre of the sweet spot
        target = (sim_low + sim_high) / 2
        score = -np.abs(sims - target)
        score[~mask] = -np.inf
        order = np.argsort(-score)[:top_k]
    out = methods.iloc[order].copy()
    out["_sim_to_gap_cluster"] = sims[order]
    return out


def generate_ideas_method_gap(
    gaps: pd.DataFrame,
    gap_embeddings: np.ndarray,
    methods: pd.DataFrame,
    method_embeddings: np.ndarray,
    cluster_labels: pd.DataFrame,
    out_tsv: Path,
    out_jsonl: Path,
    n_clusters: int | None = None,
    model: str = IDEA_MODEL,
    check_novelty: bool = True,
    k_gap_evidence: int = 5,
    k_method_evidence: int = 5,
    sim_low: float = 0.30,
    sim_high: float = 0.70,
) -> pd.DataFrame:
    """For each gap-cluster, retrieve sweet-spot method candidates and ask the
    LLM to apply one to the gaps. One idea per gap-cluster.
    """
    label_map = dict(zip(cluster_labels["cluster_id"].astype(int), cluster_labels["theme_label"]))
    cids = sorted(c for c in gaps["cluster_id"].unique() if c != -1)
    if n_clusters:
        sizes = gaps[gaps["cluster_id"].isin(cids)].groupby("cluster_id").size().sort_values(ascending=False)
        cids = sizes.head(n_clusters).index.tolist()

    client = get_llm_client()
    embedder = None
    s2 = None
    if check_novelty:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            s2 = S2Client()
        except Exception as e:
            log.warning("Disabling novelty check: %s", e)
            check_novelty = False

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    flat_rows: list[dict] = []
    with open(out_jsonl, "w", encoding="utf-8") as jf:
        for i, cid in enumerate(cids, 1):
            label = label_map.get(int(cid), "")

            gap_ev_df = _diverse_evidence(gaps, int(cid), k=k_gap_evidence)
            if gap_ev_df.empty:
                continue
            gap_ev = _evidence_payload(gap_ev_df)

            method_df = _retrieve_methods_for_cluster(
                int(cid), gap_embeddings, gaps, methods, method_embeddings,
                top_k=k_method_evidence, sim_low=sim_low, sim_high=sim_high,
            )
            if method_df.empty:
                log.warning("No methods in sweet spot for cluster %s; skipping", cid)
                continue

            method_payload = [
                {
                    "paper_id": str(r["id"]),
                    "method_type": str(r.get("method_type", "")),
                    "confidence": float(r.get("confidence", 0.0)),
                    "method_sentence": str(r["method_sentence"]),
                    "paragraph_text": str(r.get("paragraph_text", "")),
                    "similarity_to_gap_cluster": float(r["_sim_to_gap_cluster"]),
                }
                for _, r in method_df.iterrows()
            ]

            prompt = _build_method_gap_prompt(int(cid), label, gap_ev, method_payload)
            try:
                data = _call_llm_method_gap(client, prompt, model=model)
            except Exception as e:
                log.error("LLM call failed for cluster %s: %s", cid, e)
                continue
            idea = data["idea"]

            nov = {}
            if check_novelty:
                try:
                    nov = novelty_check(idea, s2, embedder, top_k=10)
                except Exception as e:
                    log.warning("Novelty check failed for cluster %s: %s", cid, e)

            closest = nov.get("closest_paper") or {}
            mean_method_sim = float(np.mean([m["similarity_to_gap_cluster"] for m in method_payload]))

            flat_rows.append({
                "mode": "method-gap",
                "cluster_a": int(cid),
                "cluster_b": None,
                "label_a": label,
                "label_b": "",
                "bridge_score": None,
                "cosine_sim": None,
                "mean_method_similarity": mean_method_sim,
                "title": idea["title"],
                "research_question": idea["research_question"],
                "method_sketch": idea["method_sketch"],
                "evaluation_plan": idea["evaluation_plan"],
                "expected_contribution": idea["expected_contribution"],
                "assumptions_and_risks": idea["assumptions_and_risks"],
                "idea_confidence": float(idea["confidence"]),
                "evidence_used_json": json.dumps(idea["evidence_used"], ensure_ascii=False),
                "novelty_score": nov.get("novelty_score"),
                "max_similarity_to_prior": nov.get("max_similarity"),
                "closest_paper_title": closest.get("title", ""),
                "closest_paper_year": closest.get("year", ""),
                "closest_paper_id": closest.get("paperId", ""),
            })

            jf.write(json.dumps({
                "mode": "method-gap",
                "pair": {"cluster_a": int(cid), "label_a": label},
                "evidence_a": gap_ev,
                "candidate_methods": method_payload,
                "prompt": prompt,
                "system": SYSTEM_METHOD_GAP,
                "model": model,
                "idea": idea,
                "novelty": nov,
            }, ensure_ascii=False) + "\n")
            log.info("  [%d/%d] cluster %d (%s) %s  mean_method_sim=%.2f  nov=%s",
                     i, len(cids), int(cid), label[:30], idea["title"][:50],
                     mean_method_sim,
                     f"{nov.get('novelty_score'):.2f}" if nov.get("novelty_score") is not None else "n/a")

    out_df = pd.DataFrame(flat_rows)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote %d method-gap ideas to %s and %s", len(out_df), out_tsv, out_jsonl)
    return out_df
