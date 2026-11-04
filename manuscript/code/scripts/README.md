# `scripts/` — dataset-building scripts

Run every script **from the repo root** (they resolve `code/src` and `data/` relative to themselves,
so the working directory must be the `em.dcs` root).

## `dataset/harvest_acl_limitations.py` — build the gold author-stated-gap corpora

Builds `data/acl_limitations.tsv` and `data/acl_futurework.tsv` from the **LimGen** dataset
(ACL-Anthology papers with author-written *mandated* "Limitations" sections; Faizullah et al.,
ECML-PKDD 2024, <https://github.com/arbmf/LimGen>, **CC BY 4.0**).

**What it does**
1. Downloads LimGen `test.jsonl` + `valid.jsonl` (~814 papers) from GitHub — keyless.
2. Limitations section → sentences → `limitation` rows.
3. Paper body → cue-gated future-work sentences (≤6 per paper) → `future_work` rows.
4. Filters: sentence sanity check, dedup, and a **leakage guard** — drops any sentence with
   token-containment ≥ 0.8 against the eval gold in `data/bench_gap/gold_sentences.tsv`.
5. Writes both TSVs (`src_id, gap_type, gap_sentence`): ~6.4k limitation + ~1.3k future-work sentences.

Only the **rule-based** helpers in `code/src/gap2idea/pipeline/{gap_prefilter,sections,gap_funnel}.py`
are used — no LLM, no GPU, no `torch`.

**Run**
```bash
# from the em.dcs repo root
pip install pandas numpy            # if not already installed  (or: cd code && uv sync)
python code/scripts/dataset/harvest_acl_limitations.py
```
- **Needs:** Python 3.10+, `pandas`, `numpy` (+ stdlib). No API key.
- **Reads:** `data/bench_gap/gold_sentences.tsv`
- **Writes:** `data/acl_limitations.tsv`, `data/acl_futurework.tsv`
- **Time:** ~1–2 min (download + sentence processing).

Output columns, sizes, and licensing: [`../../data/README.md`](../../data/README.md).
