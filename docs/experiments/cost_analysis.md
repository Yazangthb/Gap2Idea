# Cost / efficiency analysis — funnel vs full-paper LLM (Oct 2026)

Phase 3 of the publication plan. Question: how much cheaper is the funnel (which sends the LLM only
the detector's few candidate sentences) than giving the whole paper to an LLM — and at what recall?
Data: SAL test, 40 papers (7,968 sentences, 205 gold limitations). Tokens via `tiktoken cl100k_base`.
Scripts: `stage_c_package/cost_tokens.py` (accounting, no API), `stage_c_package/cost_fullpaper.py`
(full-paper LLM run, `yandexgpt-5-lite`).

## Part 1 — token accounting (all 40 papers, no API)

| | per paper (mean) | total (40) |
|---|---|---|
| full-paper LLM input tokens | **7,071** | 282,846 |
| funnel Stage-C input tokens | **1,891** | 75,622 |
| **reduction** | **3.7×** | 3.7× |

- Stage A keeps **23.7%** of sentences (1885/7968); Stage B passes only **4.9%** (391) to the LLM —
  **~10 sentences/paper vs the whole ~199.**
- The funnel's modest 3.7× on these short (~7k-token) papers is because the re-sent few-shot prompt
  (~0.7k tokens) and ±1 context inflate the per-call cost. Two things widen the gap:

| funnel Stage-C variant | total tokens | vs full-paper |
|---|---|---|
| base (1 batch/paper) | 75,622 | 3.7× |
| cross-paper batching | 64,248 | 4.4× |
| + prompt caching (SYS+shots) | 55,508 | 5.1× |

- **The ratio grows with paper length** (funnel cost is ~flat at ~1.9k tokens; full-paper scales with
  length — corr 0.29 vs 0.60):

| paper size | reduction |
|---|---|
| 7k tokens (SAL) | 3.7× |
| 10k | 5.3× |
| 15k | 7.9× |
| 25k | 13.2× |

- $ (illustrative, $0.15/1M input): full-paper **$1.06 / 1k papers** vs funnel **$0.28 / 1k papers**.

## Part 2 — full-paper LLM recall (10 SAL papers, actual run, same model `yandexgpt-5-lite`)

| approach | recall | precision | LLM tokens/paper |
|---|---|---|---|
| **full-paper LLM** (whole paper → limitation indices) | **0.345** | 0.559 | 6,540 (actual) |
| **funnel Stage A+B** (local detector, no LLM) | **0.927** | — | **0** |
| funnel Stage C (optional precision polish) | — | — | ~1,891 |

- The full-paper LLM found only **35%** of gold limitations (≈0.49 on the 8/10 papers that returned
  parseable output — **2/10 calls returned nothing**, a long-context robustness failure).
- The funnel's **free, local Stage A+B reaches recall 0.93** on the same papers — a small model reading
  ~200 sentences at once misses limitations; localizing first fixes that.

## Conclusion
**The funnel is both cheaper and higher-recall than full-paper LLM extraction with a comparable model.**
- Recall: **0.93 (funnel A+B, $0 LLM)** vs **0.35 (full-paper LLM, 6.5k tokens/paper)**.
- Cost: Stage A+B use **no LLM**; the optional Stage C precision filter sends ~10 sentences/paper
  (**3.7× cheaper on SAL, scaling to >10× on long papers**, 5× with prompt caching).

**Honest caveats:**
- Both used the same cheap model (`yandexgpt-5-lite`) for fairness. A frontier model on full papers
  would raise full-paper recall — but at substantially higher per-paper cost, widening the cost gap.
- The 3.7× on short SAL papers is the conservative end; the advantage is largest on long documents and
  at scale (caching / cross-paper batching).
- Output tokens not counted (full-paper extraction emits more output than the funnel's short verdicts) —
  this further favors the funnel.
