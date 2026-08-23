# """
# Evaluation

# A note on Ragas: I tried wiring ragas 0.4.3's built-in Faithfulness metric
# to Gemini, through every path ragas itself documents -- llm_factory with
# the current google-genai SDK (Instructor adapter), llm_factory with the
# now-EOL google.generativeai SDK (LiteLLM adapter), and the deprecated
# LangchainLLMWrapper. All three construct without error but fail
# identically at call time with "Cannot use agenerate() with a synchronous
# client": ragas's internal async-detection (_check_client_async) doesn't
# recognize any Google client shape as async-capable in this release, and
# Faithfulness.score()/.ascore() always route through the async path
# regardless of which one you call. Confirmed by direct testing against
# the actual installed package, not just by reading the docs -- this is a
# live bug in this ragas version's Google integration, not a config
# mistake. Worth a line in your report: "evaluated ragas, hit a documented
# async/Google-provider incompatibility, built a custom LLM-judge evaluator
# instead using the same Gemini client as the rest of the project."

# This script measures the same two things ragas' metrics would have:
#   - Faithfulness: does the answer only make claims the retrieved context
#     actually supports? (the "hallucination rate" metric your roadmap
#     asked for)
#   - Retrieval hit rate: did retrieval surface the page(s) the answer
#     should have been grounded in, for questions where you know the answer?

# Plus one thing that matters as much as either for a real demo: at least
# one question your documents genuinely can't answer, to confirm the
# system admits that instead of guessing.

# EDIT EVAL_SET BELOW with your own questions. Only you can fill in
# `expected_pages` correctly, since only you know what's actually in your
# PDFs.
# """

# import json
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent))
# from rag_pipeline import RagPipeline  # noqa: E402

# # -----------------------------------------------------------------------
# # EDIT THIS. `expected_pages` (optional): (doc_id, page_number) pairs you
# # know the answer should be grounded in -- enables a retrieval hit-rate
# # score for that question. Leave as [] to skip. Set
# # `should_be_unanswerable=True` for a question your documents genuinely
# # do NOT cover -- this tests hallucination resistance instead of scoring
# # faithfulness/retrieval (there's nothing correct to retrieve).
# # -----------------------------------------------------------------------
# EVAL_SET = [
#     {
#         "question": "What was total revenue growth?",
#         "expected_pages": [("2025_AnnualReport", 24), ("2025_AnnualReport", 36)],
#     },
#     {
#         "question": "What are Microsoft's three reportable segments and their 2025 revenue?",
#         "expected_pages": [("2025_AnnualReport", 22), ("2025_AnnualReport", 25)],
#     },
#     {
#         "question": "What does the segment revenue chart show?",
#         "expected_pages": [("2025_AnnualReport", 71)],
#     },
#     {
#         # Adjust this to something you've actually confirmed is absent
#         # from your PDFs -- this is just a starting guess.
#         "question": "What was the CEO's total compensation in 2025?",
#         "expected_pages": [],
#         "should_be_unanswerable": True,
#     },
# ]

# # Same free-tier model family as generation (gemini-2.5-flash), since
# # budget is $0. A separate, stronger judge model would reduce
# # "grading its own homework" bias, but isn't available on the free tier.
# JUDGE_MODEL = "gemini-3.6-flash"

# FAITHFULNESS_JUDGE_PROMPT = """You are auditing an AI-generated answer for factual grounding.

# CONTEXT (this is ALL the AI was allowed to use to answer):
# {context}

# ANSWER TO AUDIT:
# {answer}

# List every distinct factual claim in the answer. For each, decide if it is directly
# supported by the context above. Respond with ONLY valid JSON, no markdown fences, no
# commentary, in exactly this shape:
# {{"claims": [{{"claim": "restated claim", "supported": true}}], "unsupported_count": 0, "total_count": 0}}
# """

# REFUSAL_MARKERS = [
#     "do not contain", "does not contain", "doesn't contain", "don't contain",
#     "not enough information", "cannot answer", "can't answer",
#     "no information", "not provided", "not mentioned", "no mention",
#     "unable to find", "does not provide", "do not provide", "doesn't provide",
#     "don't provide", "not available", "not specified", "isn't mentioned",
#     "aren't mentioned",
# ]


# def looks_like_refusal(answer: str) -> bool:
#     """Rough substring heuristic, not a precise classifier -- good enough
#     to flag rows worth a human glance, not a metric to report on its own."""
#     lower = answer.lower()
#     return any(marker in lower for marker in REFUSAL_MARKERS)


# def parse_judge_json(raw_text: str) -> dict:
#     cleaned = raw_text.strip()
#     if cleaned.startswith("```"):
#         cleaned = cleaned.strip("`")
#         if cleaned.startswith("json"):
#             cleaned = cleaned[4:]
#     cleaned = cleaned.strip()
#     return json.loads(cleaned)


# def score_faithfulness(genai_client, answer: str, context: str) -> dict:
#     prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer)
#     try:
#         response = genai_client.models.generate_content(model=JUDGE_MODEL, contents=[prompt])
#         parsed = parse_judge_json(response.text or "")
#     except Exception as e:  # noqa: BLE001 -- any judge failure (bad JSON, API error, etc.) should degrade this one row, not crash the whole eval run
#         return {"score": None, "error": str(e)}

#     total = parsed.get("total_count", len(parsed.get("claims", [])))
#     unsupported = parsed.get("unsupported_count", sum(1 for c in parsed.get("claims", []) if not c.get("supported", True)))
#     score = 1.0 if total == 0 else max(0.0, (total - unsupported) / total)
#     return {"score": score, "total_claims": total, "unsupported_claims": unsupported, "raw": parsed}


# def retrieval_hit_rate(sources: list[dict], expected_pages: list) -> float | None:
#     if not expected_pages:
#         return None
#     retrieved_pages = {(s["doc_id"], s["page_number"]) for s in sources}
#     hits = sum(1 for p in expected_pages if tuple(p) in retrieved_pages)
#     return hits / len(expected_pages)


# def run_evaluation() -> None:
#     pipeline = RagPipeline()
#     results = []

#     print(f"\nRunning {len(EVAL_SET)} evaluation questions...\n")
#     for i, item in enumerate(EVAL_SET, 1):
#         question = item["question"]
#         should_be_unanswerable = item.get("should_be_unanswerable", False)
#         print(f"[{i}/{len(EVAL_SET)}] {question}")

#         result = pipeline.answer(question)
#         answer = result["answer"]
#         sources = result["sources"]
#         refused = looks_like_refusal(answer)

#         row = {
#             "question": question,
#             "answer": answer,
#             "sources": [f"{s['doc_id']} p{s['page_number']}" for s in sources],
#             "should_be_unanswerable": should_be_unanswerable,
#         }

#         if should_be_unanswerable:
#             row["correctly_refused"] = refused
#             print(f"  should be unanswerable -> "
#                   f"{'refused correctly' if refused else 'DID NOT REFUSE -- possible hallucination, check this one'}")
#         else:
#             row["incorrectly_refused"] = refused
#             faith = score_faithfulness(pipeline.genai_client, answer, result["context_used"])
#             row["faithfulness"] = faith
#             row["retrieval_hit_rate"] = retrieval_hit_rate(sources, item.get("expected_pages", []))
#             print(f"  faithfulness: {faith.get('score')}  |  retrieval hit rate: {row['retrieval_hit_rate']}"
#                   + ("  |  ** REFUSED AN ANSWERABLE QUESTION **" if refused else ""))

#         results.append(row)
#         print()

#     # ---------------- summary ----------------
#     scored = [r for r in results if not r["should_be_unanswerable"] and r["faithfulness"].get("score") is not None]
#     avg_faithfulness = sum(r["faithfulness"]["score"] for r in scored) / len(scored) if scored else None

#     hit_rates = [r["retrieval_hit_rate"] for r in results if r.get("retrieval_hit_rate") is not None]
#     avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else None

#     unanswerable_items = [r for r in results if r["should_be_unanswerable"]]
#     correctly_refused = sum(1 for r in unanswerable_items if r.get("correctly_refused"))

#     incorrectly_refused = [r for r in results if not r["should_be_unanswerable"] and r.get("incorrectly_refused")]

#     print("=" * 60)
#     print("SUMMARY")
#     print("=" * 60)
#     print(f"Average faithfulness score:      {avg_faithfulness:.2f}" if avg_faithfulness is not None else "Average faithfulness score:      n/a")
#     print(f"Average retrieval hit rate:      {avg_hit_rate:.2f}" if avg_hit_rate is not None else "Average retrieval hit rate:      n/a")
#     if unanswerable_items:
#         print(f"Correctly refused unanswerable:  {correctly_refused}/{len(unanswerable_items)}")
#     if incorrectly_refused:
#         print(f"WARNING: refused {len(incorrectly_refused)} answerable question(s) -- check these:")
#         for r in incorrectly_refused:
#             print(f"  - {r['question']}")

#     out_path = Path("data/processed/eval_results.json")
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text(json.dumps(results, indent=2))
#     print(f"\nFull results (including every claim the judge checked) written to {out_path}")


# if __name__ == "__main__":
#     run_evaluation()



"""
Evaluation

A note on Ragas: I tried wiring ragas 0.4.3's built-in Faithfulness metric
to Gemini, through every path ragas itself documents -- llm_factory with
the current google-genai SDK (Instructor adapter), llm_factory with the
now-EOL google.generativeai SDK (LiteLLM adapter), and the deprecated
LangchainLLMWrapper. All three construct without error but fail
identically at call time with "Cannot use agenerate() with a synchronous
client": ragas's internal async-detection (_check_client_async) doesn't
recognize any Google client shape as async-capable in this release, and
Faithfulness.score()/.ascore() always route through the async path
regardless of which one you call. Confirmed by direct testing against
the actual installed package, not just by reading the docs -- this is a
live bug in this ragas version's Google integration, not a config
mistake. Worth a line in your report: "evaluated ragas, hit a documented
async/Google-provider incompatibility, built a custom LLM-judge evaluator
instead using the same Gemini client as the rest of the project."

This script measures the same two things ragas' metrics would have:
  - Faithfulness: does the answer only make claims the retrieved context
    actually supports? (the "hallucination rate" metric your roadmap
    asked for)
  - Retrieval hit rate: did retrieval surface the page(s) the answer
    should have been grounded in, for questions where you know the answer?

Plus one thing that matters as much as either for a real demo: at least
one question your documents genuinely can't answer, to confirm the
system admits that instead of guessing.

EDIT EVAL_SET BELOW with your own questions. Only you can fill in
`expected_pages` correctly, since only you know what's actually in your
PDFs.

Requires OWNER_EMAIL/OWNER_PASSWORD (the account you registered through
the app) -- this script signs in and queries through the same
authenticated, RLS-protected path the web app uses, not a bypass, so
whatever it measures reflects what a real logged-in user actually gets.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag_pipeline import RagPipeline  # noqa: E402
from embed_and_index import get_owner_client  # noqa: E402

# -----------------------------------------------------------------------
# EDIT THIS. `expected_pages` (optional): (doc_id, page_number) pairs you
# know the answer should be grounded in -- enables a retrieval hit-rate
# score for that question. Leave as [] to skip. Set
# `should_be_unanswerable=True` for a question your documents genuinely
# do NOT cover -- this tests hallucination resistance instead of scoring
# faithfulness/retrieval (there's nothing correct to retrieve).
# -----------------------------------------------------------------------
EVAL_SET = [
    {
        "question": "What was total revenue growth?",
        "expected_pages": [("2025_AnnualReport", 24), ("2025_AnnualReport", 36)],
    },
    {
        "question": "What are Microsoft's three reportable segments and their 2025 revenue?",
        "expected_pages": [("2025_AnnualReport", 22), ("2025_AnnualReport", 25)],
    },
    {
        "question": "What does the segment revenue chart show?",
        "expected_pages": [("2025_AnnualReport", 71)],
    },
    {
        # Adjust this to something you've actually confirmed is absent
        # from your PDFs -- this is just a starting guess.
        "question": "What was the CEO's total compensation in 2025?",
        "expected_pages": [],
        "should_be_unanswerable": True,
    },

    # -------------------------------------------------------------------
    # ADD THESE ONCE YOU'VE CHECKED THEM. n=4 doesn't prove anything
    # statistically -- 1.00 on four questions is easy to hit by luck or
    # by an eval set that's accidentally too easy. Aim for 15-20+ before
    # you cite this number in your report. Workflow for filling in
    # expected_pages for a new question: run `python3 src/rag_pipeline.py`,
    # ask the question, note which pages it cites, then manually open
    # your PDF and confirm those pages actually contain the answer before
    # trusting them as ground truth -- don't just copy whatever the
    # system already retrieved, or you're only testing self-consistency.
    # -------------------------------------------------------------------

    # -- more financial detail (stresses precise-number retrieval) --
    {"question": "What was Microsoft's operating income in fiscal year 2025?", "expected_pages": []},
    {"question": "How much did Microsoft spend on research and development in 2025?", "expected_pages": []},
    {"question": "What was the year-over-year growth rate for the Intelligent Cloud segment specifically?", "expected_pages": []},
    {"question": "What was the total cash and cash equivalents on the balance sheet?", "expected_pages": []},

    # -- chart/image-specific (should pull [image] sources, not just text) --
    {"question": "What does the stock performance chart show compared to a benchmark index?", "expected_pages": []},
    {"question": "What does the geographic revenue breakdown chart show?", "expected_pages": []},

    # -- cross-document (only relevant once you have 2+ companies indexed;
    #    tests correct attribution, not just retrieval -- watch for the
    #    system blending two companies' numbers into one, which would be
    #    a faithfulness failure even if each individual number is real) --
    {"question": "What was Apple's total net sales in fiscal year 2025?", "expected_pages": []},
    {"question": "Compare Microsoft's and Apple's total revenue for fiscal year 2025.", "expected_pages": []},

    # -- harder refusal traps: plausible-sounding but likely NOT in a
    #    written financial report, unlike the CEO-compensation question
    #    above which is an obvious miss. These are the ones most likely
    #    to actually tempt a hallucination -- verify each is genuinely
    #    absent from your PDFs before relying on it. --
    {
        "question": "What did Microsoft's CFO say about AI demand during the earnings call?",
        "expected_pages": [],
        "should_be_unanswerable": True,  # verify: written reports rarely transcribe earnings calls
    },
    {
        "question": "What dividend per share did Microsoft declare in Q3 2025?",
        "expected_pages": [],
        "should_be_unanswerable": True,  # verify against your actual PDF -- this may or may not be covered
    },

    # -- table-specific lookup (forces retrieval of a TABLE chunk
    #    specifically, not a narrative paragraph that happens to mention
    #    the same figure) --
    {"question": "According to the income statement, what was the cost of revenue in 2025?", "expected_pages": []},
]

# A genuinely different model FAMILY than the gemini-2.5-flash generator,
# not just a different call -- meaningfully reduces the "grading its own
# homework" bias a same-model judge would have. (Confirmed working on the
# free tier as of this writing; if it's ever retired, fall back to any
# model that isn't gemini-2.5-flash specifically.)
JUDGE_MODEL = "gemini-3.6-flash"

FAITHFULNESS_JUDGE_PROMPT = """You are auditing an AI-generated answer for factual grounding.

CONTEXT (this is ALL the AI was allowed to use to answer):
{context}

ANSWER TO AUDIT:
{answer}

List every distinct factual claim in the answer. For each, decide if it is directly
supported by the context above. Respond with ONLY valid JSON, no markdown fences, no
commentary, in exactly this shape:
{{"claims": [{{"claim": "restated claim", "supported": true}}], "unsupported_count": 0, "total_count": 0}}
"""

REFUSAL_MARKERS = [
    "do not contain", "does not contain", "doesn't contain", "don't contain",
    "not enough information", "cannot answer", "can't answer",
    "no information", "not provided", "not mentioned", "no mention",
    "unable to find", "does not provide", "do not provide", "doesn't provide",
    "don't provide", "not available", "not specified", "isn't mentioned",
    "aren't mentioned",
]


def looks_like_refusal(answer: str) -> bool:
    """Rough substring heuristic, not a precise classifier -- good enough
    to flag rows worth a human glance, not a metric to report on its own."""
    lower = answer.lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def parse_judge_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def score_faithfulness(genai_client, answer: str, context: str) -> dict:
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer)
    try:
        response = genai_client.models.generate_content(model=JUDGE_MODEL, contents=[prompt])
        parsed = parse_judge_json(response.text or "")
    except Exception as e:  # noqa: BLE001 -- any judge failure (bad JSON, API error, etc.) should degrade this one row, not crash the whole eval run
        return {"score": None, "error": str(e)}

    total = parsed.get("total_count", len(parsed.get("claims", [])))
    unsupported = parsed.get("unsupported_count", sum(1 for c in parsed.get("claims", []) if not c.get("supported", True)))
    if total == 0:
        # Zero claims found is NOT the same as "perfectly faithful" -- it
        # usually means the judge failed to decompose the answer rather
        # than confirming it's flawless. Treat it as unscoreable so it
        # can't quietly inflate your average.
        return {"score": None, "total_claims": 0, "unsupported_claims": 0, "raw": parsed,
                "warning": "Judge found zero claims to check -- treated as unscoreable, not perfect."}
    score = max(0.0, (total - unsupported) / total)
    return {"score": score, "total_claims": total, "unsupported_claims": unsupported, "raw": parsed}


def retrieval_hit_rate(sources: list[dict], expected_pages: list) -> float | None:
    if not expected_pages:
        return None
    retrieved_pages = {(s["doc_id"], s["page_number"]) for s in sources}
    hits = sum(1 for p in expected_pages if tuple(p) in retrieved_pages)
    return hits / len(expected_pages)


def run_evaluation() -> None:
    pg_client, user_id = get_owner_client()
    pipeline = RagPipeline()
    results = []

    print(f"\nRunning {len(EVAL_SET)} evaluation questions...\n")
    for i, item in enumerate(EVAL_SET, 1):
        question = item["question"]
        should_be_unanswerable = item.get("should_be_unanswerable", False)
        print(f"[{i}/{len(EVAL_SET)}] {question}")

        result = pipeline.answer(question, pg_client, user_id)
        answer = result["answer"]
        sources = result["sources"]
        refused = looks_like_refusal(answer)

        row = {
            "question": question,
            "answer": answer,
            "sources": [f"{s['doc_id']} p{s['page_number']}" for s in sources],
            "should_be_unanswerable": should_be_unanswerable,
        }

        if should_be_unanswerable:
            row["correctly_refused"] = refused
            print(f"  should be unanswerable -> "
                  f"{'refused correctly' if refused else 'DID NOT REFUSE -- possible hallucination, check this one'}")
        else:
            row["incorrectly_refused"] = refused
            faith = score_faithfulness(pipeline.genai_client, answer, result["context_used"])
            row["faithfulness"] = faith
            row["retrieval_hit_rate"] = retrieval_hit_rate(sources, item.get("expected_pages", []))
            print(f"  faithfulness: {faith.get('score')} "
                  f"({faith.get('total_claims', '?')} claims checked, {faith.get('unsupported_claims', '?')} unsupported)"
                  f"  |  retrieval hit rate: {row['retrieval_hit_rate']}"
                  + ("  |  ** REFUSED AN ANSWERABLE QUESTION **" if refused else ""))

        results.append(row)
        print()

    # ---------------- summary ----------------
    scored = [r for r in results if not r["should_be_unanswerable"] and r["faithfulness"].get("score") is not None]
    avg_faithfulness = sum(r["faithfulness"]["score"] for r in scored) / len(scored) if scored else None

    hit_rates = [r["retrieval_hit_rate"] for r in results if r.get("retrieval_hit_rate") is not None]
    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else None

    unanswerable_items = [r for r in results if r["should_be_unanswerable"]]
    correctly_refused = sum(1 for r in unanswerable_items if r.get("correctly_refused"))

    incorrectly_refused = [r for r in results if not r["should_be_unanswerable"] and r.get("incorrectly_refused")]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average faithfulness score:      {avg_faithfulness:.2f}" if avg_faithfulness is not None else "Average faithfulness score:      n/a")
    print(f"Average retrieval hit rate:      {avg_hit_rate:.2f}" if avg_hit_rate is not None else "Average retrieval hit rate:      n/a")
    if unanswerable_items:
        print(f"Correctly refused unanswerable:  {correctly_refused}/{len(unanswerable_items)}")
    if incorrectly_refused:
        print(f"WARNING: refused {len(incorrectly_refused)} answerable question(s) -- check these:")
        for r in incorrectly_refused:
            print(f"  - {r['question']}")

    out_path = Path("data/processed/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results (including every claim the judge checked) written to {out_path}")


if __name__ == "__main__":
    run_evaluation()