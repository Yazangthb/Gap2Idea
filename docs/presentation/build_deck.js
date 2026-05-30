// ======================================================================
// Gap2Idea — thesis-defence deck, minimalist style
// (modelled after the MohSakka-presentation template the user shared)
//
// Style rules:
//   - White background everywhere
//   - Big bold black title, left-aligned, top of slide
//   - Body text in plain black, with BLUE keyword highlights inline
//   - Simple round bullets, same blue
//   - Bold page number bottom-right
//   - No cards, no shadows, no pills, no eyebrows, no decorative bars
//
// Run:
//   NODE_PATH="$(npm root -g)" node docs/presentation/build_deck.js
// ======================================================================

const pptxgen = require("pptxgenjs");
const path = require("path");

// ---------- assets (re-rasterised thesis figures) ----------
const THESIS = "C:/Users/yazan/Downloads/thesis_methodology";
const FIG = path.join(THESIS, "figs").replace(/\\/g, "/");
const FIG_PNG = path.join(THESIS, "figs_png").replace(/\\/g, "/");

const ASSETS = {
  pipeline:          `${FIG_PNG}/pipeline.png`,
  orchestrated:      `${FIG_PNG}/orchestrated_mode.png`,
  bridge:            `${FIG_PNG}/bridge_score.png`,
  benchMethodology:  `${FIG_PNG}/bench_methodology.png`,
  ablationStage1:    `${FIG}/bench/ablation_stage1.png`,
  ablationStage2:    `${FIG}/bench/ablation_stage2.png`,
  pipelineVsOracle:  `${FIG}/bench/pipeline_vs_oracle.png`,
  perPaperPipeOracle:`${FIG}/bench/per_paper_pipe_vs_oracle.png`,
  llmRecoveryVsHal:  `${FIG}/bench/llm_recovery_vs_halluc.png`,
  llmSimPerPaper:    `${FIG}/bench/llm_sim_per_paper.png`,
  regexRougePerPaper:`${FIG}/bench/regex_rouge_per_paper.png`,
  regexSectionTypes: `${FIG}/bench/regex_section_types.png`,
  summaryBars:       `${FIG}/bench/summary_bars.png`,
  metricHeatmaps:    `${FIG}/clustering_bench/metric_heatmaps.png`,
  stabilityBars:     `${FIG}/clustering_bench/stability_bars.png`,
  silhouetteNpmi:    `${FIG}/clustering_bench/silhouette_vs_npmi.png`,
  clustersGrid:      `${FIG}/clustering_bench/clusters_grid_BAAI_bge-small-en-v1.5.png`,
};

// ---------- minimalist palette ----------
const C = {
  text:    "000000",  // all body and titles
  blue:    "2962FF",  // keyword highlight + bullets
  muted:   "595959",  // for tiny secondary text
  rule:    "BFBFBF",
  ok:      "1F7A1F",
  warn:    "B83232",
};

const FONT_HEAD = "Times New Roman";
const FONT_BODY = "Times New Roman";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";   // 10 × 5.625 in
pres.author = "Yazan Alnakri";
pres.company = "BS Thesis 2024";
pres.title = "Gap2Idea — An Idea Mining Platform for Research Acceleration";

const W = 10, H = 5.625;

// ---------- minimalist helpers ----------
function pageWhite(slide) {
  slide.background = { color: "FFFFFF" };
}

// Large bold black title at top-left of the slide
function bigTitle(slide, text) {
  slide.addText(text, {
    x: 0.55, y: 0.30, w: 9.0, h: 0.85,
    fontFace: FONT_HEAD, fontSize: 36, bold: true,
    color: C.text, align: "left", valign: "top", margin: 0,
  });
}

// Bold page number bottom-right
function pageNum(slide, n) {
  slide.addText(String(n), {
    x: 9.30, y: 5.20, w: 0.50, h: 0.30,
    fontFace: FONT_BODY, fontSize: 12, bold: true,
    color: C.text, align: "right", valign: "bottom", margin: 0,
  });
}

// Helper: build a "rich text" runs array from a string with [[highlight]]
// markers. e.g. "foo [[bar]] baz" → "foo " in black, "bar" in blue, " baz".
function rich(text, opts = {}) {
  const segments = String(text).split(/(\[\[[^\]]+\]\])/g).filter(Boolean);
  return segments.map((seg) => {
    if (seg.startsWith("[[") && seg.endsWith("]]")) {
      return { text: seg.slice(2, -2), options: { color: C.blue, ...opts } };
    }
    return { text: seg, options: { color: C.text, ...opts } };
  });
}

// Bulleted list with optional [[blue]] highlights inside each item.
//
// pptxgenjs's `bullet: {code}` is paragraph-level, but applying it to a
// rich-text run whose own colour is custom (e.g. our blue) sometimes makes
// the bullet glyph disappear when rendered by LibreOffice / Google Slides.
// We sidestep the quirk by emitting bullets manually — a literal black "●"
// followed by two spaces is prepended to the first run of each item, and
// pptxgenjs's own bullet system is never used. This renders identically
// across PowerPoint, LibreOffice, and Google Slides.
function bulletList(slide, lines, x, y, w, h, opts = {}) {
  const fontSize = opts.fontSize || 16;
  const items = [];
  const TOP_BULLET = "●  ";
  const SUB_BULLET = "    ●  ";

  function pushItem(entry, isLast, indent) {
    const runs = rich(entry);
    // 1) bullet glyph always black, no paraSpaceAfter, no breakLine
    items.push({
      text: indent ? SUB_BULLET : TOP_BULLET,
      options: { color: C.text, paraSpaceAfter: 0 },
    });
    // 2) the actual content runs keep their colours
    runs.forEach((r, k) => {
      const isFinalRun = k === runs.length - 1;
      items.push({
        text: r.text,
        options: {
          ...r.options,
          breakLine: isFinalRun && !isLast,
          paraSpaceAfter: isFinalRun ? (opts.paraSpaceAfter || (indent ? 4 : 6)) : 0,
        },
      });
    });
  }

  lines.forEach((entry, i) => {
    const isLast = i === lines.length - 1;
    if (typeof entry === "string") {
      pushItem(entry, isLast, false);
    } else if (Array.isArray(entry)) {
      entry.forEach((sub, j) => {
        const isLastSub = j === entry.length - 1;
        pushItem(sub, isLast && isLastSub, true);
      });
    }
  });
  slide.addText(items, {
    x, y, w, h,
    fontFace: FONT_BODY, fontSize, valign: "top", margin: 0,
  });
}

// Plain paragraph with [[blue]] highlights (no bullet)
function para(slide, text, x, y, w, h, opts = {}) {
  slide.addText(rich(text), {
    x, y, w, h,
    fontFace: FONT_BODY, fontSize: opts.fontSize || 16,
    align: opts.align || "left", valign: opts.valign || "top",
    bold: opts.bold || false, italic: opts.italic || false,
    color: C.text, margin: 0,
  });
}

// ======================================================================
// Slides are pushed as closures; we call them at the end so the
// page-number footer has the right total.
// ======================================================================
const SLIDES = [];

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 1. Title — centred, plain
  const s = pres.addSlide(); pageWhite(s);

  s.addText(
    "Gap2Idea: An Idea Mining Platform\nfor Research Acceleration",
    {
      x: 0.5, y: 1.05, w: 9.0, h: 1.7,
      fontFace: FONT_HEAD, fontSize: 36, bold: true,
      color: C.text, align: "center", valign: "middle", margin: 0,
    }
  );

  s.addText("Bachelor Thesis Defense by:", {
    x: 0.5, y: 3.10, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 22, color: C.text,
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("Yazan Alnakri", {
    x: 0.5, y: 3.50, w: 9.0, h: 0.45,
    fontFace: FONT_BODY, fontSize: 24, bold: true, color: C.text,
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("Under the supervision of [Supervisor name]", {
    x: 0.5, y: 4.10, w: 9.0, h: 0.35,
    fontFace: FONT_BODY, fontSize: 14, color: C.text,
    align: "center", valign: "middle", margin: 0,
  });
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 2. Agenda
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Agenda");

  bulletList(s, [
    "Introduction",
    "Research Objective",
    "Related Work",
    "Proposed Pipeline",
    ["Extraction with provenance", "Theme mining + Bridge score", "Three idea-generation modes",
     "Multi-agent quality layer", "Paper drafter"],
    "Evaluation",
    "Conclusion",
  ], 0.55, 1.40, 9.0, 4.0, { fontSize: 18, paraSpaceAfter: 6 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 3. Introduction — scale of the problem
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Introduction");

  bulletList(s, [
    "More than [[100,000 ML preprints]] are uploaded to arXiv every year, and the rate keeps growing.",
    "No individual researcher can read this volume — promising directions are simply missed.",
    "The most valuable signal of what remains undone — the [[Limitations]] and [[Future Work]] sections that authors write — is scattered across thousands of PDFs.",
    "There is currently [[no production tool]] that harvests these sections at scale and turns them into actionable research ideas.",
  ], 0.55, 1.40, 9.0, 3.5, { fontSize: 17, paraSpaceAfter: 10 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 4. Introduction — the buried signal
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Introduction: the buried signal");

  bulletList(s, [
    "Authors of a paper know their own work better than any external observer.",
    "Their [[Limitations / Future Work / Open Problems]] sections are an authoritative annotation of [[\"what is still missing\"]].",
    "Yet this signal lives [[at the end]] of each PDF — rarely indexed, rarely searched, never aggregated.",
    "Consider a graduate student starting a thesis on graph neural networks: hundreds of future-work paragraphs hint at extensions — [[dynamic edges, distribution shift, multimodal]] — but they are not collectable today.",
  ], 0.55, 1.40, 9.0, 3.6, { fontSize: 16, paraSpaceAfter: 10 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 5. Research objective
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Research Objective");

  // Big centered statement — matches the template's "Research Objective" slide.
  s.addText(
    rich("Design, implement, and evaluate an end-to-end pipeline that converts a corpus of academic papers into [[clustered research gaps]], [[multi-mode idea proposals]], and conference-paper drafts — with [[every artefact traceable]] to the verbatim source text that motivated it."),
    {
      x: 1.0, y: 1.80, w: 8.0, h: 2.5,
      fontFace: FONT_BODY, fontSize: 22, color: C.text,
      align: "center", valign: "middle", margin: 0,
    }
  );

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 6. Related work
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Related Work");

  bulletList(s, [
    "[[SciMuse]] — couples knowledge graphs with LLMs to suggest broad research directions.",
    "[[CHIMERA]] — mines recombination patterns from paper abstracts.",
    "[[LiveIdeaBench]] — evaluates open-ended LLM creativity on idea-generation prompts.",
    "[[Ramón Llull Thinking Machine]] — generates ideas along predefined conceptual axes.",
    "None of them [[extract author-acknowledged gaps verbatim]], [[ground]] generated ideas in those gaps, or [[expose]] the full process to user inspection.",
  ], 0.55, 1.40, 9.0, 3.7, { fontSize: 16, paraSpaceAfter: 10 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 7. Proposed pipeline — figure + brief stage list
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Proposed Pipeline");

  // pipeline.png is portrait (2400 × 3636 ≈ 1.51 aspect h/w).
  // Place it on the right; stages text on the left.
  const imgH = 3.9;
  const imgW = imgH / (3636 / 2400);
  s.addImage({ path: ASSETS.pipeline, x: 5.6, y: 1.20, w: imgW, h: imgH });

  bulletList(s, [
    "[[Stage A]] — Acquisition and extraction.",
    "[[Stage B]] — Theme mining (cluster + label).",
    "[[Stage C]] — Idea generation in three modes.",
    "[[Stage D]] — Multi-agent evaluation.",
    "[[Stage E]] — Export and paper drafter.",
  ], 0.55, 1.40, 5.0, 3.5, { fontSize: 16, paraSpaceAfter: 10 });

  s.addText("Every artefact stays traceable to the verbatim source text.", {
    x: 0.55, y: 4.85, w: 5.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: C.muted, margin: 0,
  });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 8. Stage A — extraction
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Extraction with provenance");

  bulletList(s, [
    "[[PDF → text]] via PyMuPDF; every page read (limitations sometimes appear mid-paper).",
    "[[Three-strategy section finder]]:",
    [
      "Structured headings — numbered, Roman, or all-caps.",
      "Keyword-window fallback — \"future directions\", \"open problems\"…",
      "Tail-window fallback — last 900 tokens before References.",
    ],
    "[[Strict JSON-schema LLM call]] per paper: at most two gap items, each with [[verbatim sentence]], [[verbatim paragraph]], [[type]] ∈ {limitation, future_work, open_problem}, and a [[self-reported confidence]].",
    "Post-hoc filters: confidence ≥ 0.5, sentence length ≥ 20 chars, de-duplicate on (paper_id, sentence).",
  ], 0.55, 1.40, 9.0, 3.6, { fontSize: 14, paraSpaceAfter: 8 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 9. Stage B — theme mining
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Theme mining");

  bulletList(s, [
    "Embed every gap sentence with [[all-MiniLM-L6-v2]] (384-d, L2-normalised).",
    "Clusterer adapts to corpus size:",
    [
      "K-Means with [[silhouette-swept k]] when |F| < 150.",
      "[[HDBSCAN]] for larger corpora, noise cluster tolerated.",
    ],
    "Each cluster receives [[two labels]]:",
    [
      "[[TF-IDF keywords]] — deterministic, robust.",
      "[[LLM noun-phrase theme]] — human-readable.",
    ],
    "Both kept, because keywords are robust but stilted, and LLM labels are readable but stochastic.",
  ], 0.55, 1.40, 9.0, 3.7, { fontSize: 14, paraSpaceAfter: 6 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 10. Bridge score — figure + factors
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Bridge score");

  s.addImage({ path: ASSETS.bridge, x: 0.55, y: 1.25, w: 8.9, h: 2.0,
              sizing: { type: "contain", w: 8.9, h: 2.0 } });

  // Equation
  s.addText(
    "bridge(A, B)  =  peak(cos(μA, μB), 0.45)  ·  (1 − J(PA, PB))  ·  (0.5 + 0.5·τ(A, B))",
    {
      x: 0.55, y: 3.35, w: 8.9, h: 0.4,
      fontFace: "Consolas", fontSize: 13, bold: true, color: C.text,
      align: "center", margin: 0,
    }
  );

  bulletList(s, [
    "[[Peak at moderate similarity (0.45)]] — identical themes restate; distant themes do not combine.",
    "[[Penalise paper overlap]] — pairs drawn from the same sources go to 0.",
    "[[Reward type complementarity]] — limitations × future_work is more interesting than two limitations.",
  ], 0.55, 3.85, 8.9, 1.3, { fontSize: 13, paraSpaceAfter: 4 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 11. Three idea-generation modes
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Idea generation — three modes");

  bulletList(s, [
    "[[Within-theme]] — for each cluster, synthesise one idea from its recurring evidence.",
    "[[Bridge]] — combine two related-but-distinct themes selected by the bridge score.",
    "[[Method-gap]] — retrieve method-claim sentences from a separately-mined library, propose their application to unmet gaps (explicit [[X-solves-Y]] semantics).",
    "All three share [[one schema]] — title, research question, method sketch, evaluation plan (named metric + baseline), evidence used, confidence.",
    "Post-hoc [[evidence-overlap filter]] strips hallucinated paper IDs from the LLM's claimed evidence list.",
  ], 0.55, 1.40, 9.0, 3.6, { fontSize: 15, paraSpaceAfter: 9 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 12. Multi-agent quality layer
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Multi-agent quality layer");

  // Place figure on right
  const imgH = 3.9;
  const imgW = imgH / (3648 / 2400);
  s.addImage({ path: ASSETS.orchestrated, x: 6.0, y: 1.20, w: imgW, h: imgH });

  bulletList(s, [
    "A single LLM call cannot inspect its own output against external facts.",
    "[[Critic]] — issues live tool calls for [[novelty]] (Semantic Scholar) and [[evidence overlap]], proposes targeted revisions.",
    "[[Revisor]] — re-renders the idea addressing each issue; bounded loop (typically 2 iterations).",
    "[[Judge panel]] — independent models from [[three providers]] (Anthropic, OpenAI, Google) score on a 4-axis rubric.",
    "Cross-provider [[agreement statistic α]] surfaces consistency as an interpretable signal.",
  ], 0.55, 1.40, 5.4, 3.7, { fontSize: 13, paraSpaceAfter: 6 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 13. The critic's diagnostics
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Critic diagnostics");

  bulletList(s, [
    "[[Novelty score]]   ν = 1 − maxₕ cos(eᵢdₑₐ, eₐbₛₜrₐcₜ)",
    [
      "Embed (title + research question). Search Semantic Scholar. Cosine against top-k abstracts.",
      "Hard rule: [[ν < 0.4]] → critic must request revision.",
    ],
    "[[Evidence overlap]]   |used ∩ fed| / |used|",
    [
      "Deterministic, no LLM call. Compare each (paper_id, gap_sentence) against the fed evidence.",
      "Hard rule: [[overlap < 0.8]] → critic must request revision.",
    ],
    "Critic and synthesiser default to [[different providers]] (Claude critic, GPT synthesiser) to mitigate self-evaluation bias.",
  ], 0.55, 1.40, 9.0, 3.7, { fontSize: 14, paraSpaceAfter: 6 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 14. Judge panel
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Judge panel");

  bulletList(s, [
    "Three judges, three providers:",
    [
      "Anthropic Claude Sonnet",
      "OpenAI GPT-4o",
      "Google Gemini Flash",
    ],
    "Each judge scores on a [[1–5 Likert rubric]]:",
    [
      "[[Novelty]] — beyond rebranding existing work?",
      "[[Specificity]] — method, dataset, metric, baseline named concretely?",
      "[[Feasibility]] — could a graduate student start in two weeks?",
      "[[Evidence grounding]] — does the idea trace back to the fed gaps?",
    ],
    "[[Inter-judge agreement]]   α = 1 − σ̄ / 4   (σ̄ = mean per-axis std-dev). α = 1 ⇒ unanimous.",
  ], 0.55, 1.40, 9.0, 3.7, { fontSize: 13, paraSpaceAfter: 4 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 15. Paper drafter
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Paper drafter");

  bulletList(s, [
    "Any saved idea can be expanded by [[one further LLM call]] into a full paper plan.",
    "Output conforms to [[PAPER_DRAFT_SCHEMA]] — abstract, introduction, related work, four-subsection method, experimental setup, expected results (qualitative), discussion, conclusion.",
    "[[Three protections]] keep the draft honest:",
    [
      "Prompt forbids fabricated numbers; expected_results is marked as a [[plan]], not measurement.",
      "Post-hoc filter strips any related-work entry whose paper_id was not in the input evidence.",
      "LaTeX template renders [[\"Human work required\"]] as a visually distinct callout box.",
    ],
    "Rendering: three bundled templates (minimal / standard / IEEE) or a user-supplied custom template.",
  ], 0.55, 1.40, 9.0, 3.7, { fontSize: 13, paraSpaceAfter: 5 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 16. Evaluation: extraction quality
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Extraction quality");

  s.addImage({ path: ASSETS.ablationStage1, x: 5.20, y: 1.20, w: 4.6, h: 3.6,
              sizing: { type: "contain", w: 4.6, h: 3.6 } });

  bulletList(s, [
    "Bench: stratified sample of [[N = 100 papers]] from unarXive 2023 with labelled section spans.",
    "[[v2a]] (flat-text regex) → [[v2b]] (PDF style-aware extractor):",
    [
      "ROUGE-1 F1: [[0.322 → 0.506]] ( +57% )",
      "ROUGE-2: +97%   ·   ROUGE-L: +85%",
    ],
    "[[Same Stage-2 LLM]] both runs — the lift comes from heading recovery only.",
  ], 0.55, 1.40, 4.5, 3.6, { fontSize: 13, paraSpaceAfter: 6 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 17. Evaluation: clustering quality
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Clustering quality");

  s.addImage({ path: ASSETS.metricHeatmaps, x: 4.50, y: 1.20, w: 5.3, h: 3.7,
              sizing: { type: "contain", w: 5.3, h: 3.7 } });

  bulletList(s, [
    "[[5 clusterers × 4 encoders]] on 161 verbatim gap sentences from 98 papers.",
    "Five intrinsic metrics:",
    [
      "Silhouette ↑   Davies-Bouldin ↓   Calinski-Harabasz ↑",
      "[[NPMI ↑]] — semantic coherence proxy",
      "[[Bootstrap ARI ↑]] — stability under 80% resamples",
    ],
    "NPMI matters because the downstream consumer cares about [[theme coherence]], not geometry alone.",
  ], 0.55, 1.40, 3.8, 3.6, { fontSize: 12, paraSpaceAfter: 6 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 18. Evaluation: idea quality
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Idea quality");

  s.addText("Three-judge panel scores per axis (1–5 Likert).", {
    x: 0.55, y: 1.40, w: 9, h: 0.35,
    fontFace: FONT_BODY, fontSize: 14, color: C.text, margin: 0,
  });
  s.addText("Numbers below to be filled from idea_eval.tsv before the defence.", {
    x: 0.55, y: 1.70, w: 9, h: 0.30,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: C.warn, margin: 0,
  });

  const rows = [
    [
      { text: "Mode",         options: { bold: true, color: "FFFFFF", fill: { color: C.text }, align: "left"   } },
      { text: "Novelty",      options: { bold: true, color: "FFFFFF", fill: { color: C.text }, align: "center" } },
      { text: "Specificity",  options: { bold: true, color: "FFFFFF", fill: { color: C.text }, align: "center" } },
      { text: "Feasibility",  options: { bold: true, color: "FFFFFF", fill: { color: C.text }, align: "center" } },
      { text: "Grounding",    options: { bold: true, color: "FFFFFF", fill: { color: C.text }, align: "center" } },
      { text: "Composite",    options: { bold: true, color: "FFFFFF", fill: { color: C.text }, align: "center" } },
      { text: "α",            options: { bold: true, color: "FFFFFF", fill: { color: C.text }, align: "center" } },
    ],
    [{ text: "within-theme", options: { bold: true, align: "left" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } }],
    [{ text: "bridge", options: { bold: true, align: "left" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } }],
    [{ text: "method-gap", options: { bold: true, align: "left" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } },
     { text: "—", options: { align: "center" } }, { text: "—", options: { align: "center" } }],
  ];
  s.addTable(rows, {
    x: 0.55, y: 2.10, w: 8.9, colW: [1.6, 1.05, 1.35, 1.20, 1.30, 1.20, 1.20],
    fontFace: FONT_BODY, fontSize: 13, color: C.text,
    border: { type: "solid", pt: 0.5, color: C.rule },
  });

  s.addText(
    rich("Alongside the panel scores, the [[evidence_overlap]] is reported deterministically — catches hallucinated citations even when the panel gives 5/5."),
    {
      x: 0.55, y: 4.40, w: 8.9, h: 0.7,
      fontFace: FONT_BODY, fontSize: 13, italic: true, color: C.text,
      align: "left", valign: "top", margin: 0,
    }
  );

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 19. Contributions
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Contributions");

  bulletList(s, [
    "A [[reproducible gap-mining pipeline]] from PDF to verbatim gap sentences, with paper-level provenance.",
    "A [[bridge-score]] mechanism that ranks cluster pairs by similarity sweet-spot × paper diversity × type complementarity.",
    "[[Three idea-generation modes]] (within-theme, bridge, method-gap) sharing one schema and an evidence-overlap audit.",
    "A [[multi-agent quality layer]] — critic with tool-derived diagnostics, revisor, and cross-provider judge panel with α.",
    "A [[paper drafter]] that expands any saved idea into a structured LaTeX paper plan with a visible \"Human work required\" callout.",
  ], 0.55, 1.40, 9.0, 3.6, { fontSize: 15, paraSpaceAfter: 11 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 20. Limitations & future work
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Limitations and future work");

  bulletList(s, [
    "[[Bench scale]] — N = 100 unarXive papers is small in absolute terms; would need to grow for publication.",
    "[[No human gold standard]] for idea quality — inter-judge α serves as a proxy.",
    "Ideas can be [[noun-phrase novel]] without being experimentally novel — LLMs are plausible without being correct.",
    "Future work:",
    [
      "[[Structural graph bridges]] (edge-betweenness) over individual gaps instead of centroid pairs.",
      "An [[experimental sanity stage]] — does the method actually run on toy data?",
      "Force every idea to carry a [[falsifiable quantitative prediction]] and a named baseline.",
      "Scale to N = 1000 with author co-citation as an additional signal.",
    ],
  ], 0.55, 1.40, 9.0, 3.7, { fontSize: 14, paraSpaceAfter: 5 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 21. Conclusion
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Conclusion");

  bulletList(s, [
    "Gap2Idea harvests the [[author-acknowledged]] research gaps that every paper contains but no one aggregates.",
    "It turns them into [[evidence-grounded]] research ideas in three complementary modes, with full traceability back to the verbatim source.",
    "A [[multi-agent quality layer]] with tool-derived diagnostics and a cross-provider judge panel surfaces consistency as an interpretable signal.",
    "Any accepted idea can be expanded into a [[structured paper plan]] ready for human refinement.",
    "Released as an installable Python package with a CLI, a web dashboard, and an MCP server.",
  ], 0.55, 1.40, 9.0, 3.6, { fontSize: 15, paraSpaceAfter: 11 });

  pageNum(s, n);
});

// -----------------------------------------------------------------
SLIDES.push((n) => {
  // 22. Thank you
  const s = pres.addSlide(); pageWhite(s);
  s.addText("Thank you.", {
    x: 0.5, y: 1.80, w: 9.0, h: 1.2,
    fontFace: FONT_HEAD, fontSize: 60, bold: true, color: C.text,
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("Questions?", {
    x: 0.5, y: 3.05, w: 9.0, h: 0.6,
    fontFace: FONT_BODY, fontSize: 26, italic: true, color: C.blue,
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("github.com/Yazangthb/Gap2Idea", {
    x: 0.5, y: 4.20, w: 9.0, h: 0.4,
    fontFace: "Consolas", fontSize: 14, color: C.muted,
    align: "center", margin: 0,
  });
});

// ======================================================================
// Appendix (kept short, same minimalist style)
// ======================================================================
SLIDES.push((n) => {
  // 23. Appendix divider
  const s = pres.addSlide(); pageWhite(s);
  s.addText("Appendix", {
    x: 0.5, y: 2.30, w: 9.0, h: 1.1,
    fontFace: FONT_HEAD, fontSize: 52, bold: true, color: C.text,
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("Extra detail — bridge score, anti-hallucination, benchmarks, deployment.", {
    x: 0.5, y: 3.40, w: 9.0, h: 0.5,
    fontFace: FONT_BODY, fontSize: 16, italic: true, color: C.muted,
    align: "center", margin: 0,
  });
});

// ---------- A1 ----------
SLIDES.push((n) => {
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Bridge score — factor detail");

  s.addImage({ path: ASSETS.bridge, x: 0.55, y: 1.25, w: 8.9, h: 2.4,
              sizing: { type: "contain", w: 8.9, h: 2.4 } });

  s.addText(rich(
    "peak(s, p)  =  s/p   for s ≤ p,    (1 − s) / (1 − p)   otherwise.   J(A,B) is Jaccard overlap of source-paper sets;   τ(A,B) is the half-L₁ distance between gap-type distributions."),
    {
      x: 0.55, y: 3.85, w: 8.9, h: 1.2,
      fontFace: FONT_BODY, fontSize: 13, color: C.text,
      align: "left", valign: "top", margin: 0,
    }
  );

  pageNum(s, n);
});

// ---------- A2 ----------
SLIDES.push((n) => {
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Anti-hallucination machinery");

  bulletList(s, [
    "[[Schema enforcement]] — strict JSON-schema on every LLM call; robust parser strips markdown fences for non-OpenAI providers.",
    "[[Verbatim requirement]] — gap_sentence and paragraph_text must come verbatim from input; items < 20 chars or confidence < 0.5 dropped.",
    "[[Post-hoc evidence filter]] — evidence_used (paper_id, gap_sentence) pairs checked against fed evidence; unknown paper_ids stripped.",
    "[[Critic novelty check]] — ν = 1 − max cos(idea, S2 abstracts); ν < 0.4 forces revision.",
    "[[Critic overlap check]] — fraction of evidence_used actually fed; < 0.8 forces revision.",
    "[[Paper-drafter filter]] — drafter prompt and post-filter both restrict citations to input evidence or retrieved prior art.",
  ], 0.55, 1.40, 9.0, 3.7, { fontSize: 13, paraSpaceAfter: 4 });

  pageNum(s, n);
});

// ---------- A3 ----------
SLIDES.push((n) => {
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Bench methodology");

  s.addImage({ path: ASSETS.benchMethodology, x: 0.55, y: 1.25, w: 8.9, h: 1.9,
              sizing: { type: "contain", w: 8.9, h: 1.9 } });

  bulletList(s, [
    "Gold reference: unarXive 2023's annotated Limitations / Future Work / Open Problems spans.",
    "Stage-1: ROUGE-1, ROUGE-2, ROUGE-L F1 vs gold section text.",
    "Stage-2: [[recovery rate at τ]] and [[hallucination rate at τ]] for τ ∈ {0.5, 0.6, 0.7}.",
    "[[Oracle ceiling]] — feed gold section straight into the LLM; bounds the Stage-2 LLM in isolation.",
  ], 0.55, 3.35, 8.9, 1.7, { fontSize: 13, paraSpaceAfter: 5 });

  pageNum(s, n);
});

// ---------- A4 ----------
SLIDES.push((n) => {
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Recovery vs hallucination");

  s.addImage({ path: ASSETS.llmRecoveryVsHal, x: 0.55, y: 1.25, w: 5.6, h: 3.6,
              sizing: { type: "contain", w: 5.6, h: 3.6 } });

  bulletList(s, [
    "[[recovery(τ)]] ↑ — fraction of extracted gaps matching a gold-section sentence at cosine ≥ τ.",
    "[[halluc(τ)]] ↓ — fraction whose max cosine vs any sentence anywhere in the paper falls below τ.",
    "Together: confidence the LLM points at [[real future-work content]], not inventing it.",
  ], 6.30, 1.50, 3.4, 3.4, { fontSize: 12, paraSpaceAfter: 7 });

  pageNum(s, n);
});

// ---------- A5 ----------
SLIDES.push((n) => {
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Clustering stability");

  s.addImage({ path: ASSETS.stabilityBars, x: 0.55, y: 1.25, w: 5.6, h: 3.6,
              sizing: { type: "contain", w: 5.6, h: 3.6 } });

  bulletList(s, [
    "[[Bootstrap ARI]] over B = 10 80%-resamples per (clusterer × encoder) cell.",
    "ARI = 1 ⇒ same partition recovered; ARI = 0 ⇒ random.",
    "A clustering that [[flips between two resamples]] is not a clustering — stability is the thesis-defensible axis.",
  ], 6.30, 1.50, 3.4, 3.4, { fontSize: 12, paraSpaceAfter: 7 });

  pageNum(s, n);
});

// ---------- A6 ----------
SLIDES.push((n) => {
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Geometry vs coherence");

  s.addImage({ path: ASSETS.silhouetteNpmi, x: 0.55, y: 1.25, w: 5.6, h: 3.6,
              sizing: { type: "contain", w: 5.6, h: 3.6 } });

  bulletList(s, [
    "[[Silhouette]] measures embedding-space tightness vs separation.",
    "[[NPMI]] measures whether top tokens co-occur — proxy for human-readable theme coherence.",
    "The downstream idea generator cares about [[coherence]], not geometry alone.",
  ], 6.30, 1.50, 3.4, 3.4, { fontSize: 12, paraSpaceAfter: 7 });

  pageNum(s, n);
});

// ---------- A7 ----------
SLIDES.push((n) => {
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Deployment surfaces");

  bulletList(s, [
    "[[CLI]] — deterministic, scriptable. select-papers · extract-text · extract-sections · extract-gaps · theme-mine · generate-ideas · evaluate-ideas · export-ideas.",
    "[[Streamlit dashboard]] — six tabs to browse themes, gaps, bridge pairs, ideas, judge agreement, and paper-drafter output.",
    "[[MCP server]] — read tools (list_themes, get_theme, get_evidence, retrieve_methods, search_prior_art, score_novelty, check_evidence_overlap, list_ideas) and one write tool (save_idea), usable from Claude Desktop or any MCP-aware client.",
    "All LLM calls route through [[OpenRouter]] — a single-flag provider swap (OpenAI, Anthropic, Google, open-source).",
  ], 0.55, 1.40, 9.0, 3.7, { fontSize: 14, paraSpaceAfter: 8 });

  pageNum(s, n);
});

// ---------- A8 ----------
SLIDES.push((n) => {
  const s = pres.addSlide(); pageWhite(s);
  bigTitle(s, "Walkthrough — one idea, end-to-end");

  s.addText("Template — fill from artifacts/ideas_full.jsonl when ready.", {
    x: 0.55, y: 1.40, w: 9.0, h: 0.35,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: C.warn, margin: 0,
  });

  bulletList(s, [
    "[[Title]] — «…»",
    "[[Research question]] — «…»",
    "[[Method sketch]] — «…»",
    "[[Evaluation plan]] — «…»",
    "[[Critic]] — verdict accept, ν = «…», evidence_overlap = «…», iterations = «…»",
    "[[Judge panel]] — novelty «…», specificity «…», feasibility «…», grounding «…», α = «…»",
  ], 0.55, 1.85, 9.0, 3.3, { fontSize: 14, paraSpaceAfter: 6 });

  pageNum(s, n);
});

// ======================================================================
// Render
// ======================================================================
SLIDES.forEach((fn, i) => fn(i + 1));

// Write to a fresh filename if the canonical one is locked (PowerPoint /
// Google Drive sync). Windows-style file locks don't show up via
// fs.accessSync — we actually have to try to open the file for writing.
const fs = require("fs");
function isLocked(p) {
  try {
    if (!fs.existsSync(p)) return false;
    const fd = fs.openSync(p, "r+");
    fs.closeSync(fd);
    return false;
  } catch (e) {
    return true;
  }
}
let out = path.join(__dirname, "Gap2Idea-defense.pptx");
if (isLocked(out)) {
  out = path.join(__dirname, "Gap2Idea-defense.new.pptx");
  console.log(`(canonical file locked; writing to ${path.basename(out)} instead)`);
}
pres.writeFile({ fileName: out }).then(() => {
  console.log(`Wrote ${SLIDES.length} slides → ${out}`);
});
