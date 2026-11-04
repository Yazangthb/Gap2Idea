# Stage A on arXiv WITHOUT GROBID (PDF outline + numbered-heading + font)

arXiv gold papers: 190 (no PDF: 0). Papers where we recovered a Discussion/Conclusion/Limitations/Future heading: **182/190**.

Ceiling = recall of arXiv gold gaps in the Stage-A slice (token-containment >= 0.7); retention = sentences kept. Same existing gold sentences (gold idx preserved).

| Stage A input | ceiling recall | retention |
|---|---|---|
| flat text (current) | 40.3% | 14.9% |
| **PDF blocks (no GROBID)** | **37.5%** | 14.9% |

Recovered heading classes: other:29363, discussion:247, background:243, introduction:220, limitations:61, future_work:39

### Notes
- No Docker / no network: headings from the PDF outline (bookmarks) + numbered-section regex + strict font gating.
- `data/gold/arxiv_blocks.jsonl` is the structured full-paper input (heading/text blocks) for the scalable pipeline.
- Blocks feed `slice_section_aware(blocks=...)`; gold matched by sentence containment.
