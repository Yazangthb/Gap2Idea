import json
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

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
            "additionalProperties": False
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
                            "gap_sentence": {"type": "string"}
                        },
                        "required": ["paper_id", "gap_sentence"],
                        "additionalProperties": False
                    }
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": [
                "title",
                "research_question",
                "method_sketch",
                "evaluation_plan",
                "expected_contribution",
                "assumptions_and_risks",
                "evidence_used",
                "confidence"
            ],
            "additionalProperties": False
        }
    },
    "required": ["pair", "idea"],
    "additionalProperties": False
}

SYSTEM_IDEA = (
    "You are a research planning assistant. "
    "Use ONLY the provided evidence sentences and paragraphs. "
    "Do NOT invent datasets, results, or claims. "
    "Propose exactly one actionable research idea combining both gap themes."
)

def build_idea_prompt(cluster_a: int, cluster_b: int, label_a: str, label_b: str, ev_a: list[dict], ev_b: list[dict]) -> str:
    payload = {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "theme_a_label": label_a,
        "theme_b_label": label_b,
        "evidence_a": ev_a,
        "evidence_b": ev_b
    }
    return (
        "Return JSON matching schema.\n"
        "Constraints:\n"
        "- Use ONLY evidence_a/evidence_b content.\n"
        "- No fabricated citations, datasets, numbers, or results.\n"
        "- Method must be testable and include evaluation plan.\n"
        "- evidence_used must be a subset of provided evidence sentences.\n"
        "INPUT:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

def pick_evidence(df: pd.DataFrame, cluster_id: int, k: int = 4) -> pd.DataFrame:
    cols = ["id", "gap_type", "confidence", "gap_sentence", "paragraph_text"]
    gg = df[df["cluster_id"] == cluster_id].copy()
    gg = gg.sort_values(["confidence"], ascending=False)
    return gg[cols].head(k)

def build_evidence_payload(df: pd.DataFrame, cluster_id: int, k: int = 4) -> list[dict]:
    ev = pick_evidence(df, cluster_id, k=k)
    out = []
    for _, r in ev.iterrows():
        out.append({
            "paper_id": str(r["id"]),
            "gap_type": str(r["gap_type"]),
            "confidence": float(r["confidence"]),
            "gap_sentence": str(r["gap_sentence"]),
            "paragraph_text": str(r["paragraph_text"]),
        })
    return out

def generate_idea_for_pair(gaps: pd.DataFrame, cluster_a: int, cluster_b: int, label_a: str, label_b: str, model: str = "gpt-4o-mini") -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ev_a = build_evidence_payload(gaps, cluster_a, k=3)
    ev_b = build_evidence_payload(gaps, cluster_b, k=3)
    if len(ev_a) == 0 or len(ev_b) == 0:
        return None
    prompt = build_idea_prompt(cluster_a, cluster_b, label_a, label_b, ev_a, ev_b)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_IDEA},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "idea_synthesis", "schema": IDEA_SCHEMA, "strict": True}},
    )
    data = json.loads(resp.choices[0].message.content)
    idea = data["idea"]
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
    }