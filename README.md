# Multimodal RAG — Financial Reports

Enterprise-grade multimodal RAG over 10-K / annual report PDFs. Free and
local-first: only the final answer-generation step calls a (free-tier)
cloud API.

## Status: Phase 2 (Data Ingestion) ✅

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: smaller, faster CPU-only torch install (skip if you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

Get a free Gemini API key at aistudio.google.com (no credit card needed)
and set it as an environment variable — you'll need this in Phase 4:

```bash
export GOOGLE_API_KEY="your-key-here"     # Windows: setx GOOGLE_API_KEY "your-key-here"
```

## Getting a dataset (financial reports)

Two good free sources:

1. **Company investor-relations PDFs (recommended)** — Apple, Microsoft,
   NVIDIA, etc. post polished, professionally designed Annual
   Report PDFs directly on their investor relations sites. These are
   genuinely rich in charts, infographics, and tables — exactly what
   shows off multimodal retrieval. Search "[company] investor relations
   annual report PDF" and download 2-3.
2. **SEC EDGAR (sec.gov/edgar)** — the legal source of truth for 10-Ks.
   Filings are published as HTML, not PDF, so off-the-shelf converters
   often mangle them. The reliable free option: open the filing's
   `.htm` document directly in your browser and use Print → Save as PDF.
   Denser and less visual than IR-page PDFs, but authoritative.

Aim for **2-3 companies, ~150-300 total pages** — enough to be
genuinely multimodal without turning ingestion into an overnight job on
a laptop CPU.

Drop your PDFs into `data/raw_pdfs/`.

## Run ingestion

```bash
python3 src/ingest.py
```

This produces:
- `data/processed/text_chunks.jsonl` — per-page text + tables (tables
  are serialized to markdown, not treated as images — they retrieve far
  better as structured text)
- `data/processed/image_assets.jsonl` — metadata for every extracted image
- `data/processed/images/` — embedded raster images (photos, logos, some charts)
- `data/processed/page_renders/` — full-page screenshots of every page

**Why both `images/` and `page_renders/`?** Charts built in Excel,
PowerPoint, or matplotlib and dropped into a PDF are very often *vector*
graphics — lines and rectangles drawn directly on the page, not embedded
picture objects. Vector charts are invisible to raster image extraction;
`page.get_images()` will never find them. Rendering every page as a
bitmap is the only reliable way to guarantee a chart gets captured
regardless of how the original document was built. (I verified this
against a synthetic test PDF with a pure vector-drawn chart before
handing this off — raster extraction found zero images on that page,
the page render caught it perfectly.)

## Status: Phase 4 (Retrieval & Generation) ✅

## Run Phase 3 (embedding & indexing)

```bash
python3 src/embed_and_index.py
```

Captions come from Gemini now, not BLIP — BLIP was trained on natural
photos and produced near-useless captions on financial charts ("a table
with numbers"). If you ran the old BLIP version before, **delete
`data/processed/image_captions.jsonl`** first, or the cache will replay
the stale BLIP captions instead of calling Gemini. Expect ~10-15 minutes
for ~160 images — this is deliberate request pacing to respect the free
tier's rate limit, not a hang.

## Run Phase 4 (ask questions)

```bash
python3 src/rag_pipeline.py
```

This drops you into a simple REPL — type a question, get an answer plus
the list of sources (doc + page number) it was grounded in. Under the
hood: your question is embedded, ChromaDB is queried twice (once
filtered to text/tables, once filtered to images, so images can't get
crowded out), and the retrieved text plus the **original image files**
(not just their captions) are sent to Gemini to synthesize an answer.

Try a mix of questions to stress-test both retrieval paths, e.g.:
- "What was total revenue growth?" (should pull text/table hits)
- "What does the segment breakdown chart show?" (should pull an image hit)

If a question that should clearly hit a chart never returns an `[image]`
source, check `data/processed/image_captions.jsonl` for that page's
caption quality before assuming the retrieval logic is broken.

## Next: Phase 5 — Evaluation & UI

Coming next: a Streamlit chat interface, and Ragas metrics (faithfulness,
retrieval precision) to show hallucination/accuracy numbers in your report.

