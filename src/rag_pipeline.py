"""
Phase 4 -- Retrieval & Generation

Flow: user question -> embed it with the SAME model used at index time ->
hybrid retrieval from ChromaDB (two separate filtered queries: top text/
table chunks, and top image chunks -- run separately so images can't get
crowded out just because their captions happen to score slightly lower
than nearby text) -> hand the retrieved TEXT plus the retrieved RAW IMAGES
to Gemini, instructed to answer only from what was retrieved.

Why hand raw images to the generator instead of just their captions? The
caption's job was to be *searchable* -- similar enough to the query to be
found -- not to be a complete description. Once an image is retrieved, the
generator gets the actual pixels back, so it can read exact figures
directly off a chart instead of trusting a 2-sentence summary of it. This
is also why Phase 2 kept the original image files around instead of only
keeping captions.

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment and an
index already built by src/embed_and_index.py.
"""

# import os
# from pathlib import Path

# import chromadb
# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from google import genai
# from dotenv import load_dotenv

# load_dotenv()

# CHROMA_DIR = "data/chroma_db"
# COLLECTION_NAME = "financial_reports"
# EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# # Stronger reasoning than the flash-lite model used for bulk captioning in
# # Phase 3 -- we're now making one call per user question, not 164 in a row,
# # so we can afford the tighter free-tier limits of the better model.
# GENERATION_MODEL = "gemini-3.5-flash-lite"

# TOP_K_TEXT = 5
# TOP_K_IMAGES = 2

# SYSTEM_INSTRUCTIONS = (
#     "You are a document intelligence assistant. Answer the user's question using "
#     "ONLY the context provided below (text excerpts, tables, and images). "
#     "If the context does not contain enough information to answer, say so "
#     "explicitly rather than guessing or using outside knowledge. When you "
#     "use a specific figure, name which document and page it came from. Be "
#     "concise and precise with numbers."
# )


# class RagPipeline:
#     def __init__(self):
#         api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#         if not api_key:
#             raise RuntimeError(
#                 "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
#                 "get a free key at aistudio.google.com"
#             )
#         self.genai_client = genai.Client(api_key=api_key)

#         print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
#         self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

#         chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
#         try:
#             self.collection = chroma_client.get_collection(COLLECTION_NAME)
#         except Exception as e:
#             raise RuntimeError(
#                 f"Couldn't open collection '{COLLECTION_NAME}' in {CHROMA_DIR} -- "
#                 f"run src/embed_and_index.py first. ({e})"
#             )
#         print(f"Index loaded: {self.collection.count()} chunks available.")

#     def retrieve(self, query: str) -> dict:
#         query_embedding = self.embedder.encode(query).tolist()

#         text_results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=TOP_K_TEXT,
#             where={"content_type": {"$in": ["text", "table"]}},
#         )
#         image_results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=TOP_K_IMAGES,
#             where={"content_type": "image"},
#         )
#         return {"text": text_results, "images": image_results}

#     @staticmethod
#     def build_context_block(text_results: dict) -> str:
#         docs = text_results["documents"][0]
#         metas = text_results["metadatas"][0]
#         if not docs:
#             return "(no relevant text found)"
#         blocks = [
#             f"[{meta['content_type'].upper()} | {meta['doc_id']} p{meta['page_number']}]\n{doc}"
#             for doc, meta in zip(docs, metas)
#         ]
#         return "\n---\n".join(blocks)

#     @staticmethod
#     def load_retrieved_images(image_results: dict) -> list[tuple[Image.Image, dict]]:
#         metas = image_results["metadatas"][0]
#         loaded = []
#         for meta in metas:
#             path = meta.get("image_path", "")
#             if path and Path(path).exists():
#                 loaded.append((Image.open(path).convert("RGB"), meta))
#         return loaded

#     @staticmethod
#     def build_full_context_text(retrieved: dict) -> str:
#         """Text-only representation of everything retrieved, including a
#         stand-in for images (their captions). Used for faithfulness
#         auditing -- an important caveat: the generator actually saw the
#         raw image pixels, not just the caption, so this text view is an
#         approximation of its true context, not a perfect record of it."""
#         parts = [RagPipeline.build_context_block(retrieved["text"])]
#         img_docs = retrieved["images"]["documents"][0]
#         img_metas = retrieved["images"]["metadatas"][0]
#         if img_docs:
#             image_lines = [
#                 f"[IMAGE CAPTION | {m['doc_id']} p{m['page_number']}]\n{c}"
#                 for c, m in zip(img_docs, img_metas)
#             ]
#             parts.append("\n---\n".join(image_lines))
#         return "\n---\n".join(p for p in parts if p and p != "(no relevant text found)")

#     def answer(self, query: str) -> dict:
#         retrieved = self.retrieve(query)
#         context_block = self.build_context_block(retrieved["text"])
#         images = self.load_retrieved_images(retrieved["images"])

#         contents = [
#             SYSTEM_INSTRUCTIONS,
#             f"CONTEXT (text and tables):\n{context_block}",
#         ]
#         for image, meta in images:
#             contents.append(f"[IMAGE from {meta['doc_id']} p{meta['page_number']}]")
#             contents.append(image)
#         contents.append(f"QUESTION: {query}")

#         response = self.genai_client.models.generate_content(
#             model=GENERATION_MODEL,
#             contents=contents,
#         )

#         sources = [
#             {"doc_id": m["doc_id"], "page_number": m["page_number"], "content_type": m["content_type"]}
#             for m in retrieved["text"]["metadatas"][0]
#         ]
#         sources += [
#             {"doc_id": m["doc_id"], "page_number": m["page_number"], "content_type": "image", "image_path": m["image_path"]}
#             for _, m in images
#         ]

#         return {
#             "answer": (response.text or "").strip(),
#             "sources": sources,
#             "context_used": self.build_full_context_text(retrieved),
#         }


# if __name__ == "__main__":
#     pipeline = RagPipeline()
#     print("\nAsk a question about your documents (Ctrl+C to quit).\n")
#     while True:
#         try:
#             query = input("> ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print()
#             break
#         if not query:
#             continue

#         result = pipeline.answer(query)
#         print(f"\n{result['answer']}\n")
#         print("Sources used:")
#         for s in result["sources"]:
#             tag = f" [{s['content_type']}]" if s["content_type"] != "text" else ""
#             print(f"  - {s['doc_id']} p{s['page_number']}{tag}")
#         print()



# """
# Phase 4 -- Retrieval & Generation

# Flow: user question -> embed it with the SAME model used at index time ->
# hybrid retrieval from ChromaDB (two separate filtered queries: top text/
# table chunks, and top image chunks -- run separately so images can't get
# crowded out just because their captions happen to score slightly lower
# than nearby text) -> hand the retrieved TEXT plus the retrieved RAW IMAGES
# to Gemini, instructed to answer only from what was retrieved.

# Why hand raw images to the generator instead of just their captions? The
# caption's job was to be *searchable* -- similar enough to the query to be
# found -- not to be a complete description. Once an image is retrieved, the
# generator gets the actual pixels back, so it can read exact figures
# directly off a chart instead of trusting a 2-sentence summary of it. This
# is also why Phase 2 kept the original image files around instead of only
# keeping captions.

# Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment and an
# index already built by src/embed_and_index.py.
# """

# import os
# from pathlib import Path

# import chromadb
# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from google import genai
# from dotenv import load_dotenv
# load_dotenv()

# CHROMA_DIR = "data/chroma_db"
# COLLECTION_NAME = "financial_reports"
# EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# # Stronger reasoning than the flash-lite model used for bulk captioning in
# # Phase 3 -- we're now making one call per user question, not 164 in a row,
# # so we can afford the tighter free-tier limits of the better model.
# GENERATION_MODEL = "gemini-3.5-flash-lite"

# TOP_K_TEXT = 5
# TOP_K_IMAGES = 2

# SYSTEM_INSTRUCTIONS = (
#     "You are a document intelligence assistant. Answer the user's question using "
#     "ONLY the context provided below (text excerpts, tables, and images). "
#     "If the context does not contain enough information to answer, say so "
#     "explicitly rather than guessing or using outside knowledge. When you "
#     "use a specific figure, name which document and page it came from. Be "
#     "concise and precise with numbers."
# )


# class RagPipeline:
#     def __init__(self):
#         api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#         if not api_key:
#             raise RuntimeError(
#                 "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
#                 "get a free key at aistudio.google.com"
#             )
#         self.genai_client = genai.Client(api_key=api_key)

#         print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
#         self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

#         chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
#         try:
#             self.collection = chroma_client.get_collection(COLLECTION_NAME)
#         except Exception as e:
#             raise RuntimeError(
#                 f"Couldn't open collection '{COLLECTION_NAME}' in {CHROMA_DIR} -- "
#                 f"run src/embed_and_index.py first. ({e})"
#             )
#         print(f"Index loaded: {self.collection.count()} chunks available.")

#     def retrieve(self, query: str, user_id: str | None = None) -> dict:
#         query_embedding = self.embedder.encode(query).tolist()

#         text_where = self._build_where({"content_type": {"$in": ["text", "table"]}}, user_id)
#         image_where = self._build_where({"content_type": "image"}, user_id)

#         text_results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=TOP_K_TEXT,
#             where=text_where,
#         )
#         image_results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=TOP_K_IMAGES,
#             where=image_where,
#         )
#         return {"text": text_results, "images": image_results}

#     @staticmethod
#     def _build_where(content_filter: dict, user_id: str | None) -> dict:
#         """user_id=None means "search everything" -- used by evaluate.py,
#         which has no per-user context of its own. Every real app request
#         (from app.py) passes the logged-in user's id, which is what
#         actually enforces per-user isolation at query time -- Row Level
#         Security in Postgres protects the metadata tables, this is the
#         equivalent protection for what's retrievable from ChromaDB."""
#         if user_id is None:
#             return content_filter
#         return {"$and": [content_filter, {"user_id": user_id}]}

#     @staticmethod
#     def build_context_block(text_results: dict) -> str:
#         docs = text_results["documents"][0]
#         metas = text_results["metadatas"][0]
#         if not docs:
#             return "(no relevant text found)"
#         blocks = [
#             f"[{meta['content_type'].upper()} | {meta['doc_id']} p{meta['page_number']}]\n{doc}"
#             for doc, meta in zip(docs, metas)
#         ]
#         return "\n---\n".join(blocks)

#     @staticmethod
#     def load_retrieved_images(image_results: dict) -> list[tuple[Image.Image, dict]]:
#         metas = image_results["metadatas"][0]
#         loaded = []
#         for meta in metas:
#             path = meta.get("image_path", "")
#             if path and Path(path).exists():
#                 loaded.append((Image.open(path).convert("RGB"), meta))
#         return loaded

#     @staticmethod
#     def build_full_context_text(retrieved: dict) -> str:
#         """Text-only representation of everything retrieved, including a
#         stand-in for images (their captions). Used for faithfulness
#         auditing -- an important caveat: the generator actually saw the
#         raw image pixels, not just the caption, so this text view is an
#         approximation of its true context, not a perfect record of it."""
#         parts = [RagPipeline.build_context_block(retrieved["text"])]
#         img_docs = retrieved["images"]["documents"][0]
#         img_metas = retrieved["images"]["metadatas"][0]
#         if img_docs:
#             image_lines = [
#                 f"[IMAGE CAPTION | {m['doc_id']} p{m['page_number']}]\n{c}"
#                 for c, m in zip(img_docs, img_metas)
#             ]
#             parts.append("\n---\n".join(image_lines))
#         return "\n---\n".join(p for p in parts if p and p != "(no relevant text found)")

#     def answer(self, query: str, user_id: str | None = None) -> dict:
#         retrieved = self.retrieve(query, user_id=user_id)
#         context_block = self.build_context_block(retrieved["text"])
#         images = self.load_retrieved_images(retrieved["images"])

#         contents = [
#             SYSTEM_INSTRUCTIONS,
#             f"CONTEXT (text and tables):\n{context_block}",
#         ]
#         for image, meta in images:
#             contents.append(f"[IMAGE from {meta['doc_id']} p{meta['page_number']}]")
#             contents.append(image)
#         contents.append(f"QUESTION: {query}")

#         response = self.genai_client.models.generate_content(
#             model=GENERATION_MODEL,
#             contents=contents,
#         )

#         sources = [
#             {"doc_id": m["doc_id"], "page_number": m["page_number"], "content_type": m["content_type"]}
#             for m in retrieved["text"]["metadatas"][0]
#         ]
#         sources += [
#             {"doc_id": m["doc_id"], "page_number": m["page_number"], "content_type": "image", "image_path": m["image_path"]}
#             for _, m in images
#         ]

#         return {
#             "answer": (response.text or "").strip(),
#             "sources": sources,
#             "context_used": self.build_full_context_text(retrieved),
#         }


# if __name__ == "__main__":
#     pipeline = RagPipeline()
#     print("\nAsk a question about your financial reports (Ctrl+C to quit).\n")
#     while True:
#         try:
#             query = input("> ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print()
#             break
#         if not query:
#             continue

#         result = pipeline.answer(query)
#         print(f"\n{result['answer']}\n")
#         print("Sources used:")
#         for s in result["sources"]:
#             tag = f" [{s['content_type']}]" if s["content_type"] != "text" else ""
#             print(f"  - {s['doc_id']} p{s['page_number']}{tag}")
#         print()



"""
Retrieval & Generation

Flow: user question -> embed it with the SAME model used at index time ->
hybrid retrieval via Postgres/pgvector (two separate calls to the
match_chunks RPC -- top text/table chunks, and top image chunks, run
separately so images can't get crowded out just because their captions
score slightly lower than nearby text) -> hand the retrieved TEXT plus
the retrieved RAW IMAGES to Gemini, instructed to answer only from what
was retrieved.

This used to query ChromaDB directly. Swapped to Postgres+pgvector
(the match_chunks RPC, defined in supabase_schema.sql) as the step-4
architecture change -- see embed_and_index.py's module docstring for why.

Every retrieval call needs an authenticated Supabase client (pg_client).
Row Level Security on the chunks table means a client that's still on
the bare anon key -- or authenticated as the wrong user -- simply sees
zero rows for someone else's data, not an error. app.py builds this via
get_authenticated_client(); evaluate.py and the CLI bulk-indexer build it
via embed_and_index.get_owner_client(). This class itself never touches
auth -- it just requires the caller to hand it an already-authenticated
client, same as db_service.py's functions do.
"""

# import os
# from pathlib import Path

# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from google import genai
# from dotenv import load_dotenv

# load_dotenv()

# EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# # Stronger reasoning than the flash-lite model used for bulk captioning --
# # we're making one call per user question here, not hundreds in a row, so
# # we can afford the tighter free-tier limits of the better model.
# GENERATION_MODEL = "gemini-3.5-flash-lite"

# TOP_K_TEXT = 5
# TOP_K_IMAGES = 2

# SYSTEM_INSTRUCTIONS = (
#     "You are a document intelligence assistant. Answer the user's question using "
#     "ONLY the context provided below (text excerpts, tables, and images). "
#     "If the context does not contain enough information to answer, say so "
#     "explicitly rather than guessing or using outside knowledge. When you "
#     "use a specific figure, name which document and page it came from. Be "
#     "concise and precise with numbers."
# )


# class RagPipeline:
#     def __init__(self):
#         api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#         if not api_key:
#             raise RuntimeError(
#                 "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
#                 "get a free key at aistudio.google.com"
#             )
#         self.genai_client = genai.Client(api_key=api_key)

#         print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
#         self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

#     def retrieve(self, query: str, pg_client, user_id: str) -> dict:
#         query_embedding = self.embedder.encode(query).tolist()

#         text_response = pg_client.rpc("match_chunks", {
#             "query_embedding": query_embedding,
#             "match_user_id": user_id,
#             "content_types": ["text", "table"],
#             "match_count": TOP_K_TEXT,
#         }).execute()

#         image_response = pg_client.rpc("match_chunks", {
#             "query_embedding": query_embedding,
#             "match_user_id": user_id,
#             "content_types": ["image"],
#             "match_count": TOP_K_IMAGES,
#         }).execute()

#         return {"text": text_response.data, "images": image_response.data}

#     @staticmethod
#     def build_context_block(text_rows: list[dict]) -> str:
#         if not text_rows:
#             return "(no relevant text found)"
#         blocks = [
#             f"[{row['content_type'].upper()} | {row['filename']} p{row['page_number']}]\n{row['content']}"
#             for row in text_rows
#         ]
#         return "\n---\n".join(blocks)

#     @staticmethod
#     def load_retrieved_images(image_rows: list[dict]) -> list[tuple[Image.Image, dict]]:
#         loaded = []
#         for row in image_rows:
#             path = row.get("image_path")
#             if path and Path(path).exists():
#                 loaded.append((Image.open(path).convert("RGB"), row))
#         return loaded

#     @staticmethod
#     def build_full_context_text(retrieved: dict) -> str:
#         """Text-only representation of everything retrieved, including a
#         stand-in for images (their captions). Used for faithfulness
#         auditing in evaluate.py -- an important caveat: the generator
#         actually saw the raw image pixels, not just the caption, so this
#         text view approximates its true context rather than recording it
#         exactly."""
#         parts = [RagPipeline.build_context_block(retrieved["text"])]
#         image_rows = retrieved["images"]
#         if image_rows:
#             image_lines = [
#                 f"[IMAGE CAPTION | {row['filename']} p{row['page_number']}]\n{row['content']}"
#                 for row in image_rows
#             ]
#             parts.append("\n---\n".join(image_lines))
#         return "\n---\n".join(p for p in parts if p and p != "(no relevant text found)")

#     def answer(self, query: str, pg_client, user_id: str) -> dict:
#         retrieved = self.retrieve(query, pg_client, user_id)
#         context_block = self.build_context_block(retrieved["text"])
#         images = self.load_retrieved_images(retrieved["images"])

#         contents = [
#             SYSTEM_INSTRUCTIONS,
#             f"CONTEXT (text and tables):\n{context_block}",
#         ]
#         for image, row in images:
#             contents.append(f"[IMAGE from {row['filename']} p{row['page_number']}]")
#             contents.append(image)
#         contents.append(f"QUESTION: {query}")

#         response = self.genai_client.models.generate_content(
#             model=GENERATION_MODEL,
#             contents=contents,
#         )

#         # Keys kept as "doc_id" (not "filename") in the returned sources so
#         # app.py's existing render_sources() needs no changes for this --
#         # that wiring is part 2.
#         sources = [
#             {"doc_id": row["filename"], "page_number": row["page_number"], "content_type": row["content_type"]}
#             for row in retrieved["text"]
#         ]
#         sources += [
#             {"doc_id": row["filename"], "page_number": row["page_number"], "content_type": "image", "image_path": row["image_path"]}
#             for _, row in images
#         ]

#         return {
#             "answer": (response.text or "").strip(),
#             "sources": sources,
#             "context_used": self.build_full_context_text(retrieved),
#         }



"""
Retrieval & Generation

Flow: user question -> embed it with the SAME model used at index time ->
hybrid retrieval via Postgres/pgvector (two separate calls to the
match_chunks RPC -- top text/table chunks, and top image chunks, run
separately so images can't get crowded out just because their captions
score slightly lower than nearby text) -> hand the retrieved TEXT plus
the retrieved RAW IMAGES to Gemini, instructed to answer only from what
was retrieved.

This used to query ChromaDB directly. Swapped to Postgres+pgvector
(the match_chunks RPC, defined in supabase_schema.sql) as the step-4
architecture change -- see embed_and_index.py's module docstring for why.

Every retrieval call needs an authenticated Supabase client (pg_client).
Row Level Security on the chunks table means a client that's still on
the bare anon key -- or authenticated as the wrong user -- simply sees
zero rows for someone else's data, not an error. app.py builds this via
get_authenticated_client(); evaluate.py and the CLI bulk-indexer build it
via embed_and_index.get_owner_client(). This class itself never touches
auth -- it just requires the caller to hand it an already-authenticated
client, same as db_service.py's functions do.
"""

# import os
# from pathlib import Path

# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from google import genai
# from dotenv import load_dotenv

# load_dotenv()

# EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# # Stronger reasoning than the flash-lite model used for bulk captioning --
# # we're making one call per user question here, not hundreds in a row, so
# # we can afford the tighter free-tier limits of the better model.
# GENERATION_MODEL = "gemini-3.5-flash-lite"

# TOP_K_TEXT = 5
# TOP_K_IMAGES = 2

# SYSTEM_INSTRUCTIONS = (
#     "You are a document intelligence assistant. Answer the user's question using "
#     "ONLY the context provided below (text excerpts, tables, and images). "
#     "If the context does not contain enough information to answer, say so "
#     "explicitly rather than guessing or using outside knowledge. When you "
#     "use a specific figure, name which document and page it came from. Be "
#     "concise and precise with numbers."
# )


# class RagPipeline:
#     def __init__(self):
#         api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#         if not api_key:
#             raise RuntimeError(
#                 "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
#                 "get a free key at aistudio.google.com"
#             )
#         self.genai_client = genai.Client(api_key=api_key)

#         print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
#         self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

#     def retrieve(self, query: str, pg_client, user_id: str) -> dict:
#         query_embedding = self.embedder.encode(query).tolist()

#         text_response = pg_client.rpc("match_chunks", {
#             "query_embedding": query_embedding,
#             "match_user_id": user_id,
#             "content_types": ["text", "table"],
#             "match_count": TOP_K_TEXT,
#         }).execute()

#         image_response = pg_client.rpc("match_chunks", {
#             "query_embedding": query_embedding,
#             "match_user_id": user_id,
#             "content_types": ["image"],
#             "match_count": TOP_K_IMAGES,
#         }).execute()

#         return {"text": text_response.data, "images": image_response.data}

#     @staticmethod
#     def build_context_block(text_rows: list[dict]) -> str:
#         if not text_rows:
#             return "(no relevant text found)"
#         blocks = [
#             f"[{row['content_type'].upper()} | {row['filename']} p{row['page_number']}]\n{row['content']}"
#             for row in text_rows
#         ]
#         return "\n---\n".join(blocks)

#     @staticmethod
#     def load_retrieved_images(image_rows: list[dict]) -> list[tuple[Image.Image, dict]]:
#         loaded = []
#         for row in image_rows:
#             path = row.get("image_path")
#             if path and Path(path).exists():
#                 loaded.append((Image.open(path).convert("RGB"), row))
#         return loaded

#     @staticmethod
#     def build_full_context_text(retrieved: dict) -> str:
#         """Text-only representation of everything retrieved, including a
#         stand-in for images (their captions). Used for faithfulness
#         auditing in evaluate.py -- an important caveat: the generator
#         actually saw the raw image pixels, not just the caption, so this
#         text view approximates its true context rather than recording it
#         exactly."""
#         parts = [RagPipeline.build_context_block(retrieved["text"])]
#         image_rows = retrieved["images"]
#         if image_rows:
#             image_lines = [
#                 f"[IMAGE CAPTION | {row['filename']} p{row['page_number']}]\n{row['content']}"
#                 for row in image_rows
#             ]
#             parts.append("\n---\n".join(image_lines))
#         return "\n---\n".join(p for p in parts if p and p != "(no relevant text found)")

#     def answer(self, query: str, pg_client, user_id: str) -> dict:
#         retrieved = self.retrieve(query, pg_client, user_id)
#         context_block = self.build_context_block(retrieved["text"])
#         images = self.load_retrieved_images(retrieved["images"])

#         contents = [
#             SYSTEM_INSTRUCTIONS,
#             f"CONTEXT (text and tables):\n{context_block}",
#         ]
#         for image, row in images:
#             contents.append(f"[IMAGE from {row['filename']} p{row['page_number']}]")
#             contents.append(image)
#         contents.append(f"QUESTION: {query}")

#         response = self.genai_client.models.generate_content(
#             model=GENERATION_MODEL,
#             contents=contents,
#         )

#         # Keys kept as "doc_id" (not "filename") in the returned sources so
#         # app.py's existing render_sources() needs no changes for this --
#         # that wiring is part 2.
#         sources = [
#             {"doc_id": row["filename"], "page_number": row["page_number"], "content_type": row["content_type"]}
#             for row in retrieved["text"]
#         ]
#         sources += [
#             {"doc_id": row["filename"], "page_number": row["page_number"], "content_type": "image", "image_path": row["image_path"]}
#             for _, row in images
#         ]

#         return {
#             "answer": (response.text or "").strip(),
#             "sources": sources,
#             "context_used": self.build_full_context_text(retrieved),
#         }


# if __name__ == "__main__":
#     # Quick manual sanity-check without the full Streamlit UI. Requires
#     # OWNER_EMAIL/OWNER_PASSWORD (the account you registered through the
#     # app) -- signs in the same way the CLI bulk-indexer and evaluate.py
#     # do, so this exercises the real authenticated, RLS-protected path.
#     from embed_and_index import get_owner_client

#     pg_client, user_id = get_owner_client()
#     pipeline = RagPipeline()
#     print("\nAsk a question about your documents (Ctrl+C to quit).\n")
#     while True:
#         try:
#             query = input("> ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print()
#             break
#         if not query:
#             continue

#         result = pipeline.answer(query, pg_client, user_id)
#         print(f"\n{result['answer']}\n")
#         print("Sources used:")
#         for s in result["sources"]:
#             tag = f" [{s['content_type']}]" if s["content_type"] != "text" else ""
#             print(f"  - {s['doc_id']} p{s['page_number']}{tag}")
#         print()



"""
Retrieval & Generation

Flow: user question -> embed it with the SAME model used at index time ->
hybrid retrieval via Postgres/pgvector (two separate calls to the
match_chunks RPC -- top text/table chunks, and top image chunks, run
separately so images can't get crowded out just because their captions
score slightly lower than nearby text) -> hand the retrieved TEXT plus
the retrieved RAW IMAGES to Gemini, instructed to answer only from what
was retrieved.

This used to query ChromaDB directly. Swapped to Postgres+pgvector
(the match_chunks RPC, defined in supabase_schema.sql) as the step-4
architecture change -- see embed_and_index.py's module docstring for why.

Every retrieval call needs an authenticated Supabase client (pg_client).
Row Level Security on the chunks table means a client that's still on
the bare anon key -- or authenticated as the wrong user -- simply sees
zero rows for someone else's data, not an error. app.py builds this via
get_authenticated_client(); evaluate.py and the CLI bulk-indexer build it
via embed_and_index.get_owner_client(). This class itself never touches
auth -- it just requires the caller to hand it an already-authenticated
client, same as db_service.py's functions do.
"""

import io
import os

from PIL import Image
from sentence_transformers import SentenceTransformer
from google import genai

import storage_service
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Stronger reasoning than the flash-lite model used for bulk captioning --
# we're making one call per user question here, not hundreds in a row, so
# we can afford the tighter free-tier limits of the better model.
GENERATION_MODEL = "gemini-3.5-flash-lite"

TOP_K_TEXT = 5
TOP_K_IMAGES = 2

SYSTEM_INSTRUCTIONS = (
    "You are a document intelligence assistant. Answer the user's question using "
    "ONLY the context provided below (text excerpts, tables, and images). "
    "If the context does not contain enough information to answer, say so "
    "explicitly rather than guessing or using outside knowledge. When you "
    "use a specific figure, name which document and page it came from. Be "
    "concise and precise with numbers."
)


class RagPipeline:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this -- "
                "get a free key at aistudio.google.com"
            )
        self.genai_client = genai.Client(api_key=api_key)

        print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def retrieve(self, query: str, pg_client, user_id: str) -> dict:
        query_embedding = self.embedder.encode(query).tolist()

        text_response = pg_client.rpc("match_chunks", {
            "query_embedding": query_embedding,
            "match_user_id": user_id,
            "content_types": ["text", "table"],
            "match_count": TOP_K_TEXT,
        }).execute()

        image_response = pg_client.rpc("match_chunks", {
            "query_embedding": query_embedding,
            "match_user_id": user_id,
            "content_types": ["image"],
            "match_count": TOP_K_IMAGES,
        }).execute()

        return {"text": text_response.data, "images": image_response.data}

    @staticmethod
    def build_context_block(text_rows: list[dict]) -> str:
        if not text_rows:
            return "(no relevant text found)"
        blocks = [
            f"[{row['content_type'].upper()} | {row['filename']} p{row['page_number']}]\n{row['content']}"
            for row in text_rows
        ]
        return "\n---\n".join(blocks)

    @staticmethod
    def load_retrieved_images(image_rows: list[dict], pg_client) -> list[tuple[Image.Image, dict]]:
        """Downloads each retrieved image from Supabase Storage rather
        than reading local disk -- local files don't survive a redeploy,
        which was the entire point of the step-5 storage migration (see
        storage_service.py's module docstring). A missing/corrupted file
        is skipped rather than crashing the whole answer -- one bad image
        shouldn't take down a question that also has good text sources."""
        loaded = []
        for row in image_rows:
            storage_path = row.get("image_path")
            if not storage_path:
                continue
            try:
                file_bytes = storage_service.download_bytes(pg_client, storage_path)
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                loaded.append((image, row))
            except Exception:
                continue
        return loaded

    @staticmethod
    def build_full_context_text(retrieved: dict) -> str:
        """Text-only representation of everything retrieved, including a
        stand-in for images (their captions). Used for faithfulness
        auditing in evaluate.py -- an important caveat: the generator
        actually saw the raw image pixels, not just the caption, so this
        text view approximates its true context rather than recording it
        exactly."""
        parts = [RagPipeline.build_context_block(retrieved["text"])]
        image_rows = retrieved["images"]
        if image_rows:
            image_lines = [
                f"[IMAGE CAPTION | {row['filename']} p{row['page_number']}]\n{row['content']}"
                for row in image_rows
            ]
            parts.append("\n---\n".join(image_lines))
        return "\n---\n".join(p for p in parts if p and p != "(no relevant text found)")

    def answer(self, query: str, pg_client, user_id: str) -> dict:
        retrieved = self.retrieve(query, pg_client, user_id)
        context_block = self.build_context_block(retrieved["text"])
        images = self.load_retrieved_images(retrieved["images"], pg_client)

        contents = [
            SYSTEM_INSTRUCTIONS,
            f"CONTEXT (text and tables):\n{context_block}",
        ]
        for image, row in images:
            contents.append(f"[IMAGE from {row['filename']} p{row['page_number']}]")
            contents.append(image)
        contents.append(f"QUESTION: {query}")

        response = self.genai_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=contents,
        )

        # Keys kept as "doc_id" (not "filename") in the returned sources so
        # app.py's existing render_sources() needs no changes for this --
        # that wiring is part 2.
        sources = [
            {"doc_id": row["filename"], "page_number": row["page_number"], "content_type": row["content_type"]}
            for row in retrieved["text"]
        ]
        sources += [
            {"doc_id": row["filename"], "page_number": row["page_number"], "content_type": "image", "image_path": row["image_path"]}
            for _, row in images
        ]

        return {
            "answer": (response.text or "").strip(),
            "sources": sources,
            "context_used": self.build_full_context_text(retrieved),
        }


if __name__ == "__main__":
    # Quick manual sanity-check without the full Streamlit UI. Requires
    # OWNER_EMAIL/OWNER_PASSWORD (the account you registered through the
    # app) -- signs in the same way the CLI bulk-indexer and evaluate.py
    # do, so this exercises the real authenticated, RLS-protected path.
    from embed_and_index import get_owner_client

    pg_client, user_id = get_owner_client()
    pipeline = RagPipeline()
    print("\nAsk a question about your documents (Ctrl+C to quit).\n")
    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query:
            continue

        result = pipeline.answer(query, pg_client, user_id)
        print(f"\n{result['answer']}\n")
        print("Sources used:")
        for s in result["sources"]:
            tag = f" [{s['content_type']}]" if s["content_type"] != "text" else ""
            print(f"  - {s['doc_id']} p{s['page_number']}{tag}")
        print()