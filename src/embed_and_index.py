"""
Phase 3 -- Embedding & Indexing

1. Caption every image asset with Gemini (turns pixels into searchable text
   -- this is the "bridge" that lets a plain-text query find a relevant
   chart). We use Gemini instead of a local captioning model like BLIP
   because BLIP was trained on natural photos ("a dog on a beach") and
   produces near-useless captions on financial charts/tables ("a table
   with numbers") -- it has no real reading comprehension of what's in
   the image. Gemini can actually read the axis labels and figures.
2. Split page-level text into smaller overlapping chunks (tables stay whole)
3. Embed EVERYTHING -- text chunks, table chunks, and image captions -- with
   ONE text embedding model (bge-small-en-v1.5), so a single query vector
   can retrieve text, tables, and images side by side
4. Store in ChromaDB with metadata that lets us trace every result back to
   its source doc + page, and -- for images -- the path to the ORIGINAL
   image file, since we search over the caption but hand the real pixels
   to the vision LLM at generation time (Phase 4)

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) set in your environment --
get a free one at aistudio.google.com, no credit card needed. Captioning
164 images takes roughly 10-15 minutes on the free tier because we
deliberately pace requests to stay under the rate limit -- this is
expected, not a hang. Captions are cached to disk afterwards, so re-runs
only caption genuinely new images.

NOTE: if you already ran this with the old BLIP version, delete
data/processed/image_captions.jsonl before re-running, or the cache will
just replay the old low-quality BLIP captions instead of calling Gemini.
"""

# import json
# import os
# import time
# from pathlib import Path

# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from google import genai
# import chromadb
# from dotenv import load_dotenv
# load_dotenv()

# TEXT_CHUNKS_PATH = Path("data/processed/text_chunks.jsonl")
# IMAGE_ASSETS_PATH = Path("data/processed/image_assets.jsonl")
# CAPTIONS_CACHE_PATH = Path("data/processed/image_captions.jsonl")
# CHROMA_DIR = "data/chroma_db"
# COLLECTION_NAME = "financial_reports"

# EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# CAPTION_MODEL_NAME = "gemini-3.5-flash-lite"  # generous free-tier limits as of
#                                                 # this writing; check your live
#                                                 # quota at aistudio.google.com

# NO_VISUAL_CONTENT = "NO_VISUAL_CONTENT"
# CAPTION_PROMPT = (
#     "You are indexing a page from a company's financial report for search. "
#     "Describe this image in 2-3 sentences so someone could find it by "
#     "searching for its content. If it is a chart or graph, name the chart "
#     "type, the axis labels, and any specific numbers or trends shown. If it "
#     "is a table, summarize what it reports and 2-3 concrete figures from it. "
#     f"If it is mostly plain text with no meaningful chart, graph, or table, "
#     f"respond with exactly: {NO_VISUAL_CONTENT}"
# )
# REQUEST_DELAY_SECONDS = 4.5  # keeps us safely under a 15 requests/minute cap

# CHUNK_WORDS = 350
# CHUNK_OVERLAP = 50
# CHROMA_ADD_BATCH = 500  # keep add() calls small regardless of corpus size


# def load_jsonl(path: Path) -> list[dict]:
#     with open(path) as f:
#         return [json.loads(line) for line in f]


# def split_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
#     words = text.split()
#     if len(words) <= chunk_words:
#         return [text]
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + chunk_words
#         chunks.append(" ".join(words[start:end]))
#         if end >= len(words):
#             break
#         start = end - overlap
#     return chunks


# def build_text_documents(text_chunks: list[dict]) -> tuple[list[str], list[str], list[dict]]:
#     ids, documents, metadatas = [], [], []
#     for chunk in text_chunks:
#         if chunk["chunk_type"] == "table":
#             pieces = [chunk["content"]]  # never split a table mid-row
#         else:
#             pieces = split_text(chunk["content"])

#         for i, piece in enumerate(pieces):
#             ids.append(f"{chunk['chunk_id']}_c{i}")
#             documents.append(piece)
#             metadatas.append({
#                 "doc_id": chunk["doc_id"],
#                 "page_number": chunk["page_number"],
#                 "content_type": chunk["chunk_type"],  # "text" | "table"
#             })
#     return ids, documents, metadatas


# def caption_one_image(client: "genai.Client", image_path: str, max_retries: int = 4) -> str:
#     image = Image.open(image_path).convert("RGB")
#     for attempt in range(max_retries):
#         try:
#             response = client.models.generate_content(
#                 model=CAPTION_MODEL_NAME,
#                 contents=[CAPTION_PROMPT, image],
#             )
#             return (response.text or "").strip()
#         except Exception as e:
#             wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s
#             print(f"  ... error on {Path(image_path).name} ({e}); retrying in {wait}s")
#             time.sleep(wait)
#     print(f"  ... giving up on {Path(image_path).name} after {max_retries} retries")
#     return NO_VISUAL_CONTENT


# def caption_images(image_assets: list[dict]) -> dict[str, str]:
#     """Caption every image once; cache to disk so re-runs are instant."""
#     cached: dict[str, str] = {}
#     if CAPTIONS_CACHE_PATH.exists():
#         for row in load_jsonl(CAPTIONS_CACHE_PATH):
#             cached[row["image_id"]] = row["caption"]

#     captions = dict(cached)
#     to_caption = [a for a in image_assets if a["image_id"] not in cached]

#     if not to_caption:
#         print(f"All {len(image_assets)} images already captioned (using cache).")
#         return captions

#     api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#     if not api_key:
#         raise RuntimeError(
#             "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
#             "get a free key at aistudio.google.com"
#         )
#     client = genai.Client(api_key=api_key)

#     est_minutes = len(to_caption) * REQUEST_DELAY_SECONDS / 60
#     print(f"Captioning {len(to_caption)} new image(s) with Gemini "
#           f"(~{est_minutes:.0f} min at the paced rate -- this is expected, not a hang)...")

#     CAPTIONS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
#     with open(CAPTIONS_CACHE_PATH, "a") as f:
#         for i, asset in enumerate(to_caption):
#             caption = caption_one_image(client, asset["file_path"])
#             captions[asset["image_id"]] = caption
#             f.write(json.dumps({"image_id": asset["image_id"], "caption": caption}) + "\n")
#             if (i + 1) % 10 == 0 or (i + 1) == len(to_caption):
#                 print(f"  ... {i + 1}/{len(to_caption)} captioned")
#             time.sleep(REQUEST_DELAY_SECONDS)

#     return captions


# def build_image_documents(image_assets: list[dict], captions: dict[str, str]) -> tuple[list[str], list[str], list[dict]]:
#     ids, documents, metadatas = [], [], []
#     for asset in image_assets:
#         caption = captions.get(asset["image_id"], "")
#         if not caption:
#             continue
#         ids.append(f"img_{asset['image_id']}")
#         documents.append(caption)
#         metadatas.append({
#             "doc_id": asset["doc_id"],
#             "page_number": asset["page_number"],
#             "content_type": "image",
#             "image_source": asset["source"],   # "embedded" | "page_render"
#             "image_path": asset["file_path"],  # raw pixels for Phase 4
#         })
#     return ids, documents, metadatas


# def get_chroma_collection(fresh: bool = False):
#     """fresh=True deletes any existing collection first (used by the CLI's
#     full-corpus rebuild). fresh=False gets-or-creates without touching
#     existing data (used for incremental adds, e.g. from the Streamlit
#     uploader) -- safe to call repeatedly."""
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     if fresh:
#         existing_names = [c.name for c in client.list_collections()]
#         if COLLECTION_NAME in existing_names:
#             client.delete_collection(COLLECTION_NAME)
#     return client.get_or_create_collection(COLLECTION_NAME)


# def embed_and_upsert(collection, embedder, text_chunks: list[dict], image_assets: list[dict], captions: dict[str, str]) -> int:
#     """Embed the given chunks/captions and add them to an existing
#     collection. Never deletes anything -- safe to call incrementally."""
#     text_ids, text_docs, text_meta = build_text_documents(text_chunks)
#     img_ids, img_docs, img_meta = build_image_documents(image_assets, captions)

#     all_ids = text_ids + img_ids
#     all_docs = text_docs + img_docs
#     all_meta = text_meta + img_meta
#     if not all_ids:
#         return 0

#     embeddings = embedder.encode(all_docs, show_progress_bar=True, batch_size=32).tolist()

#     for start in range(0, len(all_ids), CHROMA_ADD_BATCH):
#         end = start + CHROMA_ADD_BATCH
#         collection.add(
#             ids=all_ids[start:end],
#             embeddings=embeddings[start:end],
#             documents=all_docs[start:end],
#             metadatas=all_meta[start:end],
#         )
#     return len(all_ids)


# def build_index() -> None:
#     text_chunks = load_jsonl(TEXT_CHUNKS_PATH)
#     image_assets = load_jsonl(IMAGE_ASSETS_PATH)
#     print(f"Loaded {len(text_chunks)} text/table page-records, {len(image_assets)} image assets")

#     captions = caption_images(image_assets)

#     print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
#     embedder = SentenceTransformer(EMBED_MODEL_NAME)

#     collection = get_chroma_collection(fresh=True)  # full rebuild from scratch

#     print("Embedding + writing to ChromaDB (this is the slow step on CPU -- grab a coffee)...")
#     n = embed_and_upsert(collection, embedder, text_chunks, image_assets, captions)

#     print(f"\nDone. Indexed {n} chunks into '{CHROMA_DIR}' (collection: '{COLLECTION_NAME}')")


# if __name__ == "__main__":
#     build_index()


"""
Phase 3 -- Embedding & Indexing

1. Caption every image asset with Gemini (turns pixels into searchable text
   -- this is the "bridge" that lets a plain-text query find a relevant
   chart). We use Gemini instead of a local captioning model like BLIP
   because BLIP was trained on natural photos ("a dog on a beach") and
   produces near-useless captions on financial charts/tables ("a table
   with numbers") -- it has no real reading comprehension of what's in
   the image. Gemini can actually read the axis labels and figures.
2. Split page-level text into smaller overlapping chunks (tables stay whole)
3. Embed EVERYTHING -- text chunks, table chunks, and image captions -- with
   ONE text embedding model (bge-small-en-v1.5), so a single query vector
   can retrieve text, tables, and images side by side
4. Store in ChromaDB with metadata that lets us trace every result back to
   its source doc + page, and -- for images -- the path to the ORIGINAL
   image file, since we search over the caption but hand the real pixels
   to the vision LLM at generation time (Phase 4)

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) set in your environment --
get a free one at aistudio.google.com, no credit card needed. Captioning
164 images takes roughly 10-15 minutes on the free tier because we
deliberately pace requests to stay under the rate limit -- this is
expected, not a hang. Captions are cached to disk afterwards, so re-runs
only caption genuinely new images.

NOTE: if you already ran this with the old BLIP version, delete
data/processed/image_captions.jsonl before re-running, or the cache will
just replay the old low-quality BLIP captions instead of calling Gemini.
"""

# import json
# import os
# import time
# from pathlib import Path

# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from google import genai
# import chromadb
# from dotenv import load_dotenv
# load_dotenv()

# TEXT_CHUNKS_PATH = Path("data/processed/text_chunks.jsonl")
# IMAGE_ASSETS_PATH = Path("data/processed/image_assets.jsonl")
# CAPTIONS_CACHE_PATH = Path("data/processed/image_captions.jsonl")
# CHROMA_DIR = "data/chroma_db"
# COLLECTION_NAME = "documents"  # renamed from "financial_reports" now that the
#                                  # product is general-purpose -- purely internal,
#                                  # but no reason to keep the old name around

# EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# CAPTION_MODEL_NAME = "gemini-3.5-flash-lite"  # generous free-tier limits as of
#                                                 # this writing; check your live
#                                                 # quota at aistudio.google.com

# NO_VISUAL_CONTENT = "NO_VISUAL_CONTENT"
# CAPTION_PROMPT = (
#     "You are indexing a page from a company's financial report for search. "
#     "Describe this image in 2-3 sentences so someone could find it by "
#     "searching for its content. If it is a chart or graph, name the chart "
#     "type, the axis labels, and any specific numbers or trends shown. If it "
#     "is a table, summarize what it reports and 2-3 concrete figures from it. "
#     f"If it is mostly plain text with no meaningful chart, graph, or table, "
#     f"respond with exactly: {NO_VISUAL_CONTENT}"
# )
# REQUEST_DELAY_SECONDS = 4.5  # keeps us safely under a 15 requests/minute cap

# CHUNK_WORDS = 350
# CHUNK_OVERLAP = 50
# CHROMA_ADD_BATCH = 500  # keep add() calls small regardless of corpus size


# def load_jsonl(path: Path) -> list[dict]:
#     with open(path) as f:
#         return [json.loads(line) for line in f]


# def split_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
#     words = text.split()
#     if len(words) <= chunk_words:
#         return [text]
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + chunk_words
#         chunks.append(" ".join(words[start:end]))
#         if end >= len(words):
#             break
#         start = end - overlap
#     return chunks


# def build_text_documents(text_chunks: list[dict], user_id: str | None = None) -> tuple[list[str], list[str], list[dict]]:
#     ids, documents, metadatas = [], [], []
#     for chunk in text_chunks:
#         if chunk["chunk_type"] == "table":
#             pieces = [chunk["content"]]  # never split a table mid-row
#         else:
#             pieces = split_text(chunk["content"])

#         for i, piece in enumerate(pieces):
#             base_id = f"{chunk['chunk_id']}_c{i}"
#             # Prefixed with user_id so two different users uploading a
#             # same-named file can't collide on the same Chroma id, which
#             # would let collection.add() silently let one user's chunk
#             # clobber another's. doc_id in the metadata stays clean (just
#             # the filename stem) since that's what's shown in source cards.
#             ids.append(f"{user_id}_{base_id}" if user_id else base_id)
#             documents.append(piece)
#             meta = {
#                 "doc_id": chunk["doc_id"],
#                 "page_number": chunk["page_number"],
#                 "content_type": chunk["chunk_type"],  # "text" | "table"
#             }
#             if user_id is not None:
#                 meta["user_id"] = user_id
#             metadatas.append(meta)
#     return ids, documents, metadatas


# def caption_one_image(client: "genai.Client", image_path: str, max_retries: int = 4) -> str:
#     image = Image.open(image_path).convert("RGB")
#     for attempt in range(max_retries):
#         try:
#             response = client.models.generate_content(
#                 model=CAPTION_MODEL_NAME,
#                 contents=[CAPTION_PROMPT, image],
#             )
#             return (response.text or "").strip()
#         except Exception as e:
#             wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s
#             print(f"  ... error on {Path(image_path).name} ({e}); retrying in {wait}s")
#             time.sleep(wait)
#     print(f"  ... giving up on {Path(image_path).name} after {max_retries} retries")
#     return NO_VISUAL_CONTENT


# def caption_images(image_assets: list[dict]) -> dict[str, str]:
#     """Caption every image once; cache to disk so re-runs are instant."""
#     cached: dict[str, str] = {}
#     if CAPTIONS_CACHE_PATH.exists():
#         for row in load_jsonl(CAPTIONS_CACHE_PATH):
#             cached[row["image_id"]] = row["caption"]

#     captions = dict(cached)
#     to_caption = [a for a in image_assets if a["image_id"] not in cached]

#     if not to_caption:
#         print(f"All {len(image_assets)} images already captioned (using cache).")
#         return captions

#     api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#     if not api_key:
#         raise RuntimeError(
#             "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
#             "get a free key at aistudio.google.com"
#         )
#     client = genai.Client(api_key=api_key)

#     est_minutes = len(to_caption) * REQUEST_DELAY_SECONDS / 60
#     print(f"Captioning {len(to_caption)} new image(s) with Gemini "
#           f"(~{est_minutes:.0f} min at the paced rate -- this is expected, not a hang)...")

#     CAPTIONS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
#     with open(CAPTIONS_CACHE_PATH, "a") as f:
#         for i, asset in enumerate(to_caption):
#             caption = caption_one_image(client, asset["file_path"])
#             captions[asset["image_id"]] = caption
#             f.write(json.dumps({"image_id": asset["image_id"], "caption": caption}) + "\n")
#             if (i + 1) % 10 == 0 or (i + 1) == len(to_caption):
#                 print(f"  ... {i + 1}/{len(to_caption)} captioned")
#             time.sleep(REQUEST_DELAY_SECONDS)

#     return captions


# def build_image_documents(image_assets: list[dict], captions: dict[str, str], user_id: str | None = None) -> tuple[list[str], list[str], list[dict]]:
#     ids, documents, metadatas = [], [], []
#     for asset in image_assets:
#         caption = captions.get(asset["image_id"], "")
#         if not caption:
#             continue
#         base_id = f"img_{asset['image_id']}"
#         ids.append(f"{user_id}_{base_id}" if user_id else base_id)
#         documents.append(caption)
#         meta = {
#             "doc_id": asset["doc_id"],
#             "page_number": asset["page_number"],
#             "content_type": "image",
#             "image_source": asset["source"],   # "embedded" | "page_render"
#             "image_path": asset["file_path"],  # raw pixels for Phase 4
#         }
#         if user_id is not None:
#             meta["user_id"] = user_id
#         metadatas.append(meta)
#     return ids, documents, metadatas


# def get_chroma_collection(fresh: bool = False):
#     """fresh=True deletes any existing collection first (used by the CLI's
#     full-corpus rebuild). fresh=False gets-or-creates without touching
#     existing data (used for incremental adds, e.g. from the Streamlit
#     uploader) -- safe to call repeatedly."""
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     if fresh:
#         existing_names = [c.name for c in client.list_collections()]
#         if COLLECTION_NAME in existing_names:
#             client.delete_collection(COLLECTION_NAME)
#     return client.get_or_create_collection(COLLECTION_NAME)


# def embed_and_upsert(collection, embedder, text_chunks: list[dict], image_assets: list[dict], captions: dict[str, str], user_id: str | None = None) -> int:
#     """Embed the given chunks/captions and add them to an existing
#     collection. Never deletes anything -- safe to call incrementally."""
#     text_ids, text_docs, text_meta = build_text_documents(text_chunks, user_id=user_id)
#     img_ids, img_docs, img_meta = build_image_documents(image_assets, captions, user_id=user_id)

#     all_ids = text_ids + img_ids
#     all_docs = text_docs + img_docs
#     all_meta = text_meta + img_meta
#     if not all_ids:
#         return 0

#     embeddings = embedder.encode(all_docs, show_progress_bar=True, batch_size=32).tolist()

#     for start in range(0, len(all_ids), CHROMA_ADD_BATCH):
#         end = start + CHROMA_ADD_BATCH
#         collection.add(
#             ids=all_ids[start:end],
#             embeddings=embeddings[start:end],
#             documents=all_docs[start:end],
#             metadatas=all_meta[start:end],
#         )
#     return len(all_ids)


# def build_index() -> None:
#     # One-time migration note: your existing corpus was indexed before
#     # per-user filtering existed, so none of those chunks carry a
#     # user_id -- they'd become invisible now that retrieval filters by
#     # it. Set OWNER_USER_ID (your own Supabase user id, from the app
#     # sidebar or the Supabase dashboard's Authentication -> Users table)
#     # before running this once, and the full rebuild below re-tags
#     # everything as yours. Image captions are cached, so this is fast --
#     # only the embedding step re-runs, not the Gemini captioning.
#     owner_user_id = os.environ.get("OWNER_USER_ID")
#     if not owner_user_id:
#         print("WARNING: OWNER_USER_ID is not set -- indexing without a user_id. "
#               "Chunks indexed this way won't be visible to any logged-in user "
#               "once retrieval filtering is in place. See the comment in build_index().")

#     text_chunks = load_jsonl(TEXT_CHUNKS_PATH)
#     image_assets = load_jsonl(IMAGE_ASSETS_PATH)
#     print(f"Loaded {len(text_chunks)} text/table page-records, {len(image_assets)} image assets")

#     captions = caption_images(image_assets)

#     print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
#     embedder = SentenceTransformer(EMBED_MODEL_NAME)

#     collection = get_chroma_collection(fresh=True)  # full rebuild from scratch

#     print("Embedding + writing to ChromaDB (this is the slow step on CPU -- grab a coffee)...")
#     n = embed_and_upsert(collection, embedder, text_chunks, image_assets, captions, user_id=owner_user_id)

#     print(f"\nDone. Indexed {n} chunks into '{CHROMA_DIR}' (collection: '{COLLECTION_NAME}')")


# if __name__ == "__main__":
#     build_index()


"""
Embedding & Indexing

1. Caption every image asset with Gemini (turns pixels into searchable text
   -- this is the "bridge" that lets a plain-text query find a relevant
   chart). We use Gemini instead of a local captioning model like BLIP
   because BLIP was trained on natural photos ("a dog on a beach") and
   produces near-useless captions on charts/tables ("a table with
   numbers") -- it has no real reading comprehension of what's in the
   image. Gemini can actually read the axis labels and figures.
2. Split page-level text into smaller overlapping chunks (tables stay whole)
3. Embed EVERYTHING -- text chunks, table chunks, and image captions -- with
   ONE text embedding model (bge-small-en-v1.5), so a single query vector
   can retrieve text, tables, and images side by side
4. Store in Postgres (Supabase, pgvector extension) with a document_id
   foreign key, tagged with user_id, and with the path to the ORIGINAL
   image file for content_type='image' rows, since we search over the
   caption but hand the real pixels to the vision LLM at generation time.

This used to store into ChromaDB. Swapped to Postgres+pgvector because
(a) ChromaDB's local files don't survive a redeploy on free hosting, and
(b) a document_id foreign key with ON DELETE CASCADE means deleting a
document automatically removes its chunks -- verified locally against a
real Postgres+pgvector instance, including that Row Level Security
genuinely blocks cross-user reads AND rejects writes that try to forge
another user's user_id, not just that the SQL parses.

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) set in your environment --
get a free one at aistudio.google.com, no credit card needed. Captioning
many images takes a while on the free tier because we deliberately pace
requests to stay under the rate limit -- this is expected, not a hang.
Captions are cached to disk afterwards, so re-runs only caption
genuinely new images.
"""

# import json
# import os
# import sys
# import time
# from pathlib import Path
# from dataclasses import asdict

# from PIL import Image
# from google import genai
# from dotenv import load_dotenv
# load_dotenv()


# sys.path.insert(0, str(Path(__file__).parent))
# from ingest import extract_text_and_images, extract_tables  # noqa: E402
# import db_service  # noqa: E402

# CAPTIONS_CACHE_PATH = Path("data/processed/image_captions.jsonl")

# EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# CAPTION_MODEL_NAME = "gemini-3.5-flash-lite"  # generous free-tier limits as of
#                                                 # this writing; check your live
#                                                 # quota at aistudio.google.com

# NO_VISUAL_CONTENT = "NO_VISUAL_CONTENT"
# CAPTION_PROMPT = (
#     "You are indexing a page from a document for search. "
#     "Describe this image in 2-3 sentences so someone could find it by "
#     "searching for its content. If it is a chart or graph, name the chart "
#     "type, the axis labels, and any specific numbers or trends shown. If it "
#     "is a table, summarize what it reports and 2-3 concrete figures from it. "
#     f"If it is mostly plain text with no meaningful chart, graph, or table, "
#     f"respond with exactly: {NO_VISUAL_CONTENT}"
# )
# REQUEST_DELAY_SECONDS = 4.5  # keeps us safely under a 15 requests/minute cap

# CHUNK_WORDS = 350
# CHUNK_OVERLAP = 50
# CHUNK_INSERT_BATCH = 200  # conservative batch size for PostgREST insert payloads


# def load_jsonl(path: Path) -> list[dict]:
#     with open(path) as f:
#         return [json.loads(line) for line in f]


# def split_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
#     words = text.split()
#     if len(words) <= chunk_words:
#         return [text]
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + chunk_words
#         chunks.append(" ".join(words[start:end]))
#         if end >= len(words):
#             break
#         start = end - overlap
#     return chunks


# def caption_one_image(client: "genai.Client", image_path: str, max_retries: int = 4) -> str:
#     image = Image.open(image_path).convert("RGB")
#     for attempt in range(max_retries):
#         try:
#             response = client.models.generate_content(
#                 model=CAPTION_MODEL_NAME,
#                 contents=[CAPTION_PROMPT, image],
#             )
#             return (response.text or "").strip()
#         except Exception as e:
#             wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s
#             print(f"  ... error on {Path(image_path).name} ({e}); retrying in {wait}s")
#             time.sleep(wait)
#     print(f"  ... giving up on {Path(image_path).name} after {max_retries} retries")
#     return NO_VISUAL_CONTENT


# def caption_images(image_assets: list[dict]) -> dict[str, str]:
#     """Caption every image once; cache to disk so re-runs are instant."""
#     cached: dict[str, str] = {}
#     if CAPTIONS_CACHE_PATH.exists():
#         for row in load_jsonl(CAPTIONS_CACHE_PATH):
#             cached[row["image_id"]] = row["caption"]

#     captions = dict(cached)
#     to_caption = [a for a in image_assets if a["image_id"] not in cached]

#     if not to_caption:
#         print(f"All {len(image_assets)} images already captioned (using cache).")
#         return captions

#     api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#     if not api_key:
#         raise RuntimeError(
#             "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
#             "get a free key at aistudio.google.com"
#         )
#     client = genai.Client(api_key=api_key)

#     est_minutes = len(to_caption) * REQUEST_DELAY_SECONDS / 60
#     print(f"Captioning {len(to_caption)} new image(s) with Gemini "
#           f"(~{est_minutes:.0f} min at the paced rate -- this is expected, not a hang)...")

#     CAPTIONS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
#     with open(CAPTIONS_CACHE_PATH, "a") as f:
#         for i, asset in enumerate(to_caption):
#             caption = caption_one_image(client, asset["file_path"])
#             captions[asset["image_id"]] = caption
#             f.write(json.dumps({"image_id": asset["image_id"], "caption": caption}) + "\n")
#             if (i + 1) % 10 == 0 or (i + 1) == len(to_caption):
#                 print(f"  ... {i + 1}/{len(to_caption)} captioned")
#             time.sleep(REQUEST_DELAY_SECONDS)

#     return captions


# def build_chunk_rows(text_chunks: list[dict], image_assets: list[dict], captions: dict[str, str],
#                       user_id: str, document_id: str) -> tuple[list[dict], list[str]]:
#     """Returns (row dicts without embeddings yet, content strings to embed)
#     -- kept separate from the embedding/DB-insert step so this part stays
#     trivially unit-testable without a live model or database."""
#     rows = []
#     for chunk in text_chunks:
#         pieces = [chunk["content"]] if chunk["chunk_type"] == "table" else split_text(chunk["content"])
#         for piece in pieces:
#             rows.append({
#                 "document_id": document_id,
#                 "user_id": user_id,
#                 "page_number": chunk["page_number"],
#                 "content_type": chunk["chunk_type"],  # "text" | "table"
#                 "content": piece,
#                 "image_path": None,
#             })

#     for asset in image_assets:
#         caption = captions.get(asset["image_id"], "").strip()
#         # Skip images the captioner decided have no real chart/table/graph
#         # content -- indexing every plain-text page render as a "visual"
#         # chunk would just pollute retrieval with noise. (This check was
#         # previously missing entirely -- NO_VISUAL_CONTENT strings were
#         # silently being indexed as if they were real captions.)
#         if not caption or caption == NO_VISUAL_CONTENT:
#             continue
#         rows.append({
#             "document_id": document_id,
#             "user_id": user_id,
#             "page_number": asset["page_number"],
#             "content_type": "image",
#             "content": caption,
#             "image_path": asset["file_path"],
#         })

#     contents = [r["content"] for r in rows]
#     return rows, contents


# def embed_and_upsert(pg_client, embedder, text_chunks: list[dict], image_assets: list[dict],
#                       captions: dict[str, str], user_id: str, document_id: str) -> int:
#     rows, contents = build_chunk_rows(text_chunks, image_assets, captions, user_id, document_id)
#     if not rows:
#         return 0

#     embeddings = embedder.encode(contents, show_progress_bar=True, batch_size=32).tolist()
#     for row, embedding in zip(rows, embeddings):
#         row["embedding"] = embedding

#     for start in range(0, len(rows), CHUNK_INSERT_BATCH):
#         pg_client.table("chunks").insert(rows[start:start + CHUNK_INSERT_BATCH]).execute()

#     return len(rows)


# def process_and_index_pdf(pdf_path: Path, pg_client, embedder, user_id: str) -> dict:
#     """Extract, caption, embed, and store one PDF end-to-end: creates its
#     documents row, tracks status through processing, inserts its chunks.
#     Shared by both the CLI bulk-indexer (build_index, below) and the
#     Streamlit upload handler (app.py) -- exactly one place this logic
#     lives, so the two can't quietly drift apart."""
#     filename = pdf_path.name

#     if db_service.get_document_by_filename(pg_client, user_id, filename) is not None:
#         return {"status": "skipped", "filename": filename, "reason": "already indexed"}

#     doc_row = db_service.create_document(pg_client, user_id, filename, pdf_path.stat().st_size)

#     try:
#         db_service.update_document(pg_client, doc_row["id"], status="processing")

#         text_objs, image_objs = extract_text_and_images(pdf_path)
#         table_objs = extract_tables(pdf_path)
#         text_chunks = [asdict(c) for c in text_objs + table_objs]
#         image_assets = [asdict(a) for a in image_objs]
#         page_count = len({c.page_number for c in text_objs}) if text_objs else 0

#         captions = caption_images(image_assets)  # only calls Gemini for genuinely new images
#         n_added = embed_and_upsert(pg_client, embedder, text_chunks, image_assets, captions, user_id, doc_row["id"])

#         db_service.update_document(
#             pg_client, doc_row["id"],
#             status="completed", page_count=page_count, chunk_count=n_added, image_count=len(image_assets),
#         )
#         return {"status": "completed", "filename": filename, "document_id": doc_row["id"], "chunks": n_added}

#     except Exception as e:
#         db_service.update_document(pg_client, doc_row["id"], status="failed", error_message=str(e)[:500])
#         raise


# def get_owner_client():
#     """Signs in as whichever account OWNER_EMAIL/OWNER_PASSWORD point to
#     (the account you registered through the app) and returns an
#     authenticated client plus that user's id. The CLI indexer uses the
#     exact same auth path a real user would -- no special bypass -- so it
#     exercises the same RLS-protected code path the web app uses."""
#     import auth_service
#     email = os.environ.get("OWNER_EMAIL")
#     password = os.environ.get("OWNER_PASSWORD")
#     if not email or not password:
#         raise RuntimeError(
#             "Set OWNER_EMAIL and OWNER_PASSWORD (the account you registered "
#             "through the app) before running this."
#         )
#     client = auth_service.get_supabase_client()
#     result = auth_service.sign_in(client, email, password)
#     if not result.success or not result.access_token:
#         raise RuntimeError(f"Sign-in failed: {result.message}")
#     client.auth.set_session(result.access_token, result.refresh_token)
#     return client, result.user_id


# def build_index() -> None:
#     """Bulk CLI indexer -- processes every PDF in data/raw_pdfs/ that
#     isn't already indexed for the owner account. Useful for loading a
#     batch of documents at once instead of uploading one at a time
#     through the sidebar."""
#     from sentence_transformers import SentenceTransformer

#     pg_client, user_id = get_owner_client()

#     pdf_files = sorted(Path("data/raw_pdfs").glob("*.pdf"))
#     if not pdf_files:
#         print("No PDFs found in data/raw_pdfs/ -- drop some in there and re-run.")
#         return

#     print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
#     embedder = SentenceTransformer(EMBED_MODEL_NAME)

#     for pdf_path in pdf_files:
#         print(f"\nProcessing {pdf_path.name}...")
#         result = process_and_index_pdf(pdf_path, pg_client, embedder, user_id)
#         print(f"  -> {result}")


# if __name__ == "__main__":
#     build_index()



"""
Embedding & Indexing

1. Caption every image asset with Gemini (turns pixels into searchable text
   -- this is the "bridge" that lets a plain-text query find a relevant
   chart). We use Gemini instead of a local captioning model like BLIP
   because BLIP was trained on natural photos ("a dog on a beach") and
   produces near-useless captions on charts/tables ("a table with
   numbers") -- it has no real reading comprehension of what's in the
   image. Gemini can actually read the axis labels and figures.
2. Split page-level text into smaller overlapping chunks (tables stay whole)
3. Embed EVERYTHING -- text chunks, table chunks, and image captions -- with
   ONE text embedding model (bge-small-en-v1.5), so a single query vector
   can retrieve text, tables, and images side by side
4. Store in Postgres (Supabase, pgvector extension) with a document_id
   foreign key, tagged with user_id, and with the path to the ORIGINAL
   image file for content_type='image' rows, since we search over the
   caption but hand the real pixels to the vision LLM at generation time.

This used to store into ChromaDB. Swapped to Postgres+pgvector because
(a) ChromaDB's local files don't survive a redeploy on free hosting, and
(b) a document_id foreign key with ON DELETE CASCADE means deleting a
document automatically removes its chunks -- verified locally against a
real Postgres+pgvector instance, including that Row Level Security
genuinely blocks cross-user reads AND rejects writes that try to forge
another user's user_id, not just that the SQL parses.

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) set in your environment --
get a free one at aistudio.google.com, no credit card needed. Captioning
many images takes a while on the free tier because we deliberately pace
requests to stay under the rate limit -- this is expected, not a hang.
Captions are cached to disk afterwards, so re-runs only caption
genuinely new images.
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import asdict

from PIL import Image
from google import genai
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from ingest import extract_text_and_images, extract_tables  # noqa: E402
import db_service  # noqa: E402
import storage_service  # noqa: E402

CAPTIONS_CACHE_PATH = Path("data/processed/image_captions.jsonl")

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CAPTION_MODEL_NAME = "gemini-3.5-flash-lite"  # generous free-tier limits as of
                                                # this writing; check your live
                                                # quota at aistudio.google.com

NO_VISUAL_CONTENT = "NO_VISUAL_CONTENT"
CAPTION_PROMPT = (
    "You are indexing a page from a document for search. "
    "Describe this image in 2-3 sentences so someone could find it by "
    "searching for its content. If it is a chart or graph, name the chart "
    "type, the axis labels, and any specific numbers or trends shown. If it "
    "is a table, summarize what it reports and 2-3 concrete figures from it. "
    f"If it is mostly plain text with no meaningful chart, graph, or table, "
    f"respond with exactly: {NO_VISUAL_CONTENT}"
)
REQUEST_DELAY_SECONDS = 4.5  # keeps us safely under a 15 requests/minute cap

CHUNK_WORDS = 350
CHUNK_OVERLAP = 50
CHUNK_INSERT_BATCH = 200  # conservative batch size for PostgREST insert payloads


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def split_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def caption_one_image(client: "genai.Client", image_path: str, max_retries: int = 4) -> str:
    image = Image.open(image_path).convert("RGB")
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=CAPTION_MODEL_NAME,
                contents=[CAPTION_PROMPT, image],
            )
            return (response.text or "").strip()
        except Exception as e:
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s
            print(f"  ... error on {Path(image_path).name} ({e}); retrying in {wait}s")
            time.sleep(wait)
    print(f"  ... giving up on {Path(image_path).name} after {max_retries} retries")
    return NO_VISUAL_CONTENT


def caption_images(image_assets: list[dict]) -> dict[str, str]:
    """Caption every image once; cache to disk so re-runs are instant."""
    cached: dict[str, str] = {}
    if CAPTIONS_CACHE_PATH.exists():
        for row in load_jsonl(CAPTIONS_CACHE_PATH):
            cached[row["image_id"]] = row["caption"]

    captions = dict(cached)
    to_caption = [a for a in image_assets if a["image_id"] not in cached]

    if not to_caption:
        print(f"All {len(image_assets)} images already captioned (using cache).")
        return captions

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
            "get a free key at aistudio.google.com"
        )
    client = genai.Client(api_key=api_key)

    est_minutes = len(to_caption) * REQUEST_DELAY_SECONDS / 60
    print(f"Captioning {len(to_caption)} new image(s) with Gemini "
          f"(~{est_minutes:.0f} min at the paced rate -- this is expected, not a hang)...")

    CAPTIONS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_CACHE_PATH, "a") as f:
        for i, asset in enumerate(to_caption):
            caption = caption_one_image(client, asset["file_path"])
            captions[asset["image_id"]] = caption
            f.write(json.dumps({"image_id": asset["image_id"], "caption": caption}) + "\n")
            if (i + 1) % 10 == 0 or (i + 1) == len(to_caption):
                print(f"  ... {i + 1}/{len(to_caption)} captioned")
            time.sleep(REQUEST_DELAY_SECONDS)

    return captions


def build_chunk_rows(text_chunks: list[dict], image_assets: list[dict], captions: dict[str, str],
                      user_id: str, document_id: str) -> tuple[list[dict], list[str]]:
    """Returns (row dicts without embeddings yet, content strings to embed)
    -- kept separate from the embedding/DB-insert step so this part stays
    trivially unit-testable without a live model or database."""
    rows = []
    for chunk in text_chunks:
        pieces = [chunk["content"]] if chunk["chunk_type"] == "table" else split_text(chunk["content"])
        for piece in pieces:
            rows.append({
                "document_id": document_id,
                "user_id": user_id,
                "page_number": chunk["page_number"],
                "content_type": chunk["chunk_type"],  # "text" | "table"
                "content": piece,
                "image_path": None,
            })

    for asset in image_assets:
        caption = captions.get(asset["image_id"], "").strip()
        # Skip images the captioner decided have no real chart/table/graph
        # content -- indexing every plain-text page render as a "visual"
        # chunk would just pollute retrieval with noise. (This check was
        # previously missing entirely -- NO_VISUAL_CONTENT strings were
        # silently being indexed as if they were real captions.)
        if not caption or caption == NO_VISUAL_CONTENT:
            continue
        rows.append({
            "document_id": document_id,
            "user_id": user_id,
            "page_number": asset["page_number"],
            "content_type": "image",
            "content": caption,
            "image_path": asset["file_path"],
        })

    contents = [r["content"] for r in rows]
    return rows, contents


def embed_and_upsert(pg_client, embedder, text_chunks: list[dict], image_assets: list[dict],
                      captions: dict[str, str], user_id: str, document_id: str) -> int:
    rows, contents = build_chunk_rows(text_chunks, image_assets, captions, user_id, document_id)
    if not rows:
        return 0

    embeddings = embedder.encode(contents, show_progress_bar=True, batch_size=32).tolist()
    for row, embedding in zip(rows, embeddings):
        row["embedding"] = embedding

    for start in range(0, len(rows), CHUNK_INSERT_BATCH):
        pg_client.table("chunks").insert(rows[start:start + CHUNK_INSERT_BATCH]).execute()

    return len(rows)


def process_and_index_pdf(pdf_path: Path, pg_client, embedder, user_id: str) -> dict:
    """Extract, caption, embed, and store one PDF end-to-end: creates its
    documents row, tracks status through processing, uploads the PDF and
    every extracted image to Supabase Storage, and inserts its chunks.
    Shared by both the CLI bulk-indexer (build_index, below) and the
    Streamlit upload handler (app.py) -- exactly one place this logic
    lives, so the two can't quietly drift apart.

    Ordering matters here: images are captioned WHILE their file_path
    still points to local disk (Gemini captioning opens the file with
    PIL, which can't read a remote storage key), and only rewritten to
    their Storage path afterward, right before that path gets persisted
    into chunks.image_path. Caption first, upload+rewrite second -- doing
    it the other way silently breaks captioning with a file-not-found."""
    filename = pdf_path.name

    if db_service.get_document_by_filename(pg_client, user_id, filename) is not None:
        return {"status": "skipped", "filename": filename, "reason": "already indexed"}

    doc_row = db_service.create_document(pg_client, user_id, filename, pdf_path.stat().st_size)
    document_id = doc_row["id"]

    try:
        db_service.update_document(pg_client, document_id, status="processing")

        # Upload the raw PDF right away -- if anything below fails, the
        # original file is still safely persisted for a retry, not lost
        # along with the failed attempt.
        pdf_path_in_storage = storage_service.upload_pdf(pg_client, user_id, document_id, pdf_path.read_bytes())
        db_service.update_document(pg_client, document_id, storage_path=pdf_path_in_storage)

        text_objs, image_objs = extract_text_and_images(pdf_path)
        table_objs = extract_tables(pdf_path)
        text_chunks = [asdict(c) for c in text_objs + table_objs]
        image_assets = [asdict(a) for a in image_objs]
        page_count = len({c.page_number for c in text_objs}) if text_objs else 0

        # Caption FIRST, while file_path is still a local path (see
        # docstring above for why this ordering is load-bearing).
        captions = caption_images(image_assets)

        # NOW upload each extracted image and rewrite its file_path to
        # the Storage path -- this is what embed_and_upsert persists as
        # chunks.image_path, and what load_retrieved_images downloads
        # from at query time instead of reading local disk.
        for asset in image_assets:
            local_image_path = Path(asset["file_path"])
            asset["file_path"] = storage_service.upload_image(pg_client, user_id, document_id, local_image_path)

        n_added = embed_and_upsert(pg_client, embedder, text_chunks, image_assets, captions, user_id, document_id)

        db_service.update_document(
            pg_client, document_id,
            status="completed", page_count=page_count, chunk_count=n_added, image_count=len(image_assets),
        )
        return {"status": "completed", "filename": filename, "document_id": document_id, "chunks": n_added}

    except Exception as e:
        db_service.update_document(pg_client, document_id, status="failed", error_message=str(e)[:500])
        raise


def get_owner_client():
    """Signs in as whichever account OWNER_EMAIL/OWNER_PASSWORD point to
    (the account you registered through the app) and returns an
    authenticated client plus that user's id. The CLI indexer uses the
    exact same auth path a real user would -- no special bypass -- so it
    exercises the same RLS-protected code path the web app uses."""
    import auth_service
    email = os.environ.get("OWNER_EMAIL")
    password = os.environ.get("OWNER_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "Set OWNER_EMAIL and OWNER_PASSWORD (the account you registered "
            "through the app) before running this."
        )
    client = auth_service.get_supabase_client()
    result = auth_service.sign_in(client, email, password)
    if not result.success or not result.access_token:
        raise RuntimeError(f"Sign-in failed: {result.message}")
    client.auth.set_session(result.access_token, result.refresh_token)
    return client, result.user_id


def build_index() -> None:
    """Bulk CLI indexer -- processes every PDF in data/raw_pdfs/ that
    isn't already indexed for the owner account. Useful for loading a
    batch of documents at once instead of uploading one at a time
    through the sidebar."""
    from sentence_transformers import SentenceTransformer

    pg_client, user_id = get_owner_client()

    pdf_files = sorted(Path("data/raw_pdfs").glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/raw_pdfs/ -- drop some in there and re-run.")
        return

    print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    for pdf_path in pdf_files:
        print(f"\nProcessing {pdf_path.name}...")
        result = process_and_index_pdf(pdf_path, pg_client, embedder, user_id)
        print(f"  -> {result}")


if __name__ == "__main__":
    build_index()