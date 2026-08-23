# """
# DocuMind AI -- Streamlit UI

# General-purpose multimodal document intelligence: chat interface over the
# RAG pipeline built in Phases 2-4, plus a sidebar that lets you add new PDFs
# to the index without leaving the app. Works over any structured PDF --
# research papers, technical docs, manuals, reports -- not just financial
# filings. Uploading reuses the exact same ingest -> caption -> embed
# functions as the CLI scripts, just scoped to the one new file, so it never
# touches or rebuilds your existing index.

# Run with:
#     streamlit run app.py
# """

# import html
# import sys
# from dataclasses import asdict
# from pathlib import Path

# import streamlit as st

# sys.path.insert(0, str(Path(__file__).parent / "src"))

# from rag_pipeline import RagPipeline  # noqa: E402
# from ingest import extract_text_and_images, extract_tables  # noqa: E402
# from embed_and_index import caption_images, embed_and_upsert  # noqa: E402
# from auth_service import get_supabase_client, sign_up, sign_in, sign_out  # noqa: E402
# import db_service  # noqa: E402

# APP_NAME = "DocuMind AI"
# APP_TAGLINE = "Ask questions. Understand documents. Get grounded answers."
# SUGGESTED_QUESTIONS = [
#     "What are the key findings?",
#     "Summarize this document.",
#     "What does this chart show?",
#     "Compare the main metrics.",
# ]

# st.set_page_config(page_title=APP_NAME, page_icon="\U0001F9E0", layout="wide")

# # ---------------- Design system ----------------
# # Streamlit's own theme.toml drives native widgets (buttons, inputs, chat
# # bubbles); this CSS layer only styles the custom elements below it
# # (source cards, welcome state) that Streamlit has no built-in component
# # for. Deliberately uses soft/translucent borders and shadows rather than
# # hardcoded hex greys, so it degrades reasonably in both light and dark
# # mode instead of looking wrong in whichever one wasn't hand-tuned.
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

# html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

# .brand-header { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.1rem; }
# .brand-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
# .brand-tagline { color: rgba(120,120,130,0.9); font-size: 0.95rem; margin-bottom: 1.2rem; }

# .welcome-state { text-align: center; padding: 3rem 1rem 1.5rem 1rem; }
# .welcome-state h1 { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.5rem; }
# .welcome-state p { color: rgba(120,120,130,0.95); font-size: 1.05rem; max-width: 32rem; margin: 0 auto; }
# .suggested-label {
#     text-align: center; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em;
#     text-transform: uppercase; color: rgba(120,120,130,0.8); margin: 1.8rem 0 0.6rem 0;
# }

# .source-card {
#     border: 1px solid rgba(120,120,130,0.22);
#     border-radius: 10px;
#     padding: 0.55rem 0.85rem;
#     margin-bottom: 0.4rem;
#     box-shadow: 0 1px 2px rgba(0,0,0,0.04);
#     font-size: 0.88rem;
# }
# .source-card .doc-name { font-weight: 600; }
# .source-card .badge {
#     display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
#     text-transform: uppercase; padding: 0.05rem 0.5rem; border-radius: 999px;
#     background: rgba(79,70,229,0.12); color: #4F46E5; margin-left: 0.5rem;
# }
# </style>
# """, unsafe_allow_html=True)


# @st.cache_resource(show_spinner="Loading index and embedding model...")
# def load_pipeline() -> RagPipeline:
#     return RagPipeline()


# # ---------------- Authentication gate ----------------
# # Runs before anything else -- the embedding model and Chroma connection
# # below are only loaded once someone is actually logged in.
# for _key in ("auth_user_id", "auth_email", "auth_access_token", "auth_refresh_token"):
#     if _key not in st.session_state:
#         st.session_state[_key] = None

# if not st.session_state.auth_access_token:
#     st.markdown(
#         f'<div class="brand-header"><span style="font-size:1.7rem;">\U0001F9E0</span><h1>{APP_NAME}</h1></div>'
#         f'<div class="brand-tagline">{APP_TAGLINE}</div>',
#         unsafe_allow_html=True,
#     )
#     try:
#         _auth_client = get_supabase_client()
#     except RuntimeError as e:
#         st.error(str(e))
#         st.stop()

#     login_tab, register_tab = st.tabs(["Log in", "Create account"])

#     with login_tab:
#         with st.form("login_form"):
#             login_email = st.text_input("Email", key="login_email")
#             login_password = st.text_input("Password", type="password", key="login_password")
#             login_submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
#         if login_submitted:
#             result = sign_in(_auth_client, login_email, login_password)
#             if result.success and result.access_token:
#                 st.session_state.auth_user_id = result.user_id
#                 st.session_state.auth_email = result.email
#                 st.session_state.auth_access_token = result.access_token
#                 st.session_state.auth_refresh_token = result.refresh_token
#                 st.rerun()
#             else:
#                 st.error(result.message)

#     with register_tab:
#         with st.form("register_form"):
#             reg_email = st.text_input("Email", key="register_email")
#             reg_password = st.text_input("Password", type="password", key="register_password",
#                                           help="At least 6 characters")
#             register_submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
#         if register_submitted:
#             result = sign_up(_auth_client, reg_email, reg_password)
#             if result.success and result.access_token:
#                 st.session_state.auth_user_id = result.user_id
#                 st.session_state.auth_email = result.email
#                 st.session_state.auth_access_token = result.access_token
#                 st.session_state.auth_refresh_token = result.refresh_token
#                 st.rerun()
#             elif result.success:
#                 st.info(result.message)  # created, but email confirmation is pending
#             else:
#                 st.error(result.message)

#     st.stop()


# def get_authenticated_client():
#     """Fresh Supabase client hydrated with the CURRENT user's session.
#     Deliberately NOT wrapped in @st.cache_resource -- that cache is
#     shared across every user of the app, so caching an authenticated
#     client would leak one user's session to everyone else. Constructing
#     a client is cheap (no network call happens until .execute()), so
#     there's no real cost to building it fresh each time it's needed."""
#     client = get_supabase_client()
#     client.auth.set_session(st.session_state.auth_access_token, st.session_state.auth_refresh_token)
#     return client


# def process_uploaded_pdf(uploaded_file, pipeline: RagPipeline, pg_client, user_id: str):
#     """Ingest + caption + embed one uploaded PDF, tagged with the
#     uploader's user_id so retrieval stays scoped to them. Returns the
#     number of chunks added, or None if this filename is already indexed
#     for THIS user (per-user now, not global -- two different users can
#     upload a same-named file independently). Raises on genuine failure,
#     after recording it in Postgres so it shows up as "failed" in the
#     document list instead of silently vanishing."""
#     filename = uploaded_file.name

#     if db_service.get_document_by_filename(pg_client, user_id, filename) is not None:
#         return None

#     doc_row = db_service.create_document(pg_client, user_id, filename, len(uploaded_file.getvalue()))

#     try:
#         db_service.update_document(pg_client, doc_row["id"], status="processing")

#         raw_dir = Path("data/raw_pdfs")
#         raw_dir.mkdir(parents=True, exist_ok=True)
#         pdf_path = raw_dir / filename
#         pdf_path.write_bytes(uploaded_file.getvalue())

#         text_objs, image_objs = extract_text_and_images(pdf_path)
#         table_objs = extract_tables(pdf_path)
#         text_chunks = [asdict(c) for c in text_objs + table_objs]
#         image_assets = [asdict(a) for a in image_objs]
#         page_count = len({c.page_number for c in text_objs}) if text_objs else 0

#         captions = caption_images(image_assets)  # only calls Gemini for genuinely new images
#         n_added = embed_and_upsert(
#             pipeline.collection, pipeline.embedder, text_chunks, image_assets, captions, user_id=user_id
#         )

#         db_service.update_document(
#             pg_client, doc_row["id"],
#             status="completed", page_count=page_count,
#             chunk_count=n_added, image_count=len(image_assets),
#         )
#         return n_added

#     except Exception as e:
#         db_service.update_document(pg_client, doc_row["id"], status="failed", error_message=str(e)[:500])
#         raise


# def render_sources(sources: list[dict]) -> None:
#     text_sources = [s for s in sources if s["content_type"] != "image"]
#     image_sources = [s for s in sources if s["content_type"] == "image"]

#     with st.expander(f"Sources ({len(sources)})"):
#         if text_sources:
#             seen = set()
#             for s in text_sources:
#                 key = (s["doc_id"], s["page_number"], s["content_type"])
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 doc_name = html.escape(str(s["doc_id"]))
#                 st.markdown(
#                     f'<div class="source-card">'
#                     f'<span class="doc-name">{doc_name}</span> &middot; page {s["page_number"]}'
#                     f'<span class="badge">{html.escape(s["content_type"])}</span>'
#                     f'</div>',
#                     unsafe_allow_html=True,
#                 )

#         if image_sources:
#             st.markdown("**Images used**")
#             seen_paths = set()
#             unique_images = []
#             for s in image_sources:
#                 path = s.get("image_path")
#                 if path and path not in seen_paths and Path(path).exists():
#                     seen_paths.add(path)
#                     unique_images.append(s)
#             cols = st.columns(min(len(unique_images), 3) or 1)
#             for i, s in enumerate(unique_images):
#                 with cols[i % len(cols)]:
#                     st.image(s["image_path"], caption=f"{s['doc_id']} p{s['page_number']}", use_container_width=True)


# # ---------------- Sidebar ----------------
# with st.sidebar:
#     st.markdown(f'<div class="brand-header"><span style="font-size:1.3rem;">\U0001F9E0</span>'
#                 f'<span style="font-weight:700;font-size:1.1rem;">{APP_NAME}</span></div>', unsafe_allow_html=True)
#     st.caption(APP_TAGLINE)
#     st.divider()

#     st.subheader("\U0001F4C1 Document Library")

#     pipeline = None
#     pg_client = None
#     user_documents = []
#     try:
#         pipeline = load_pipeline()
#         pg_client = get_authenticated_client()
#         user_documents = db_service.list_documents(pg_client, st.session_state.auth_user_id)

#         if not user_documents:
#             st.caption("No documents yet -- add one below.")
#         for doc in user_documents:
#             with st.container(border=True):
#                 st.markdown(f"**{html.escape(doc['filename'])}**")
#                 status = doc["status"]
#                 status_emoji = {"completed": "\u2705", "processing": "\u23F3", "failed": "\u274C", "uploading": "\u23F3"}.get(status, "\u2022")
#                 st.caption(f"{status_emoji} {status} &middot; {doc.get('chunk_count', 0)} chunks &middot; "
#                            f"{doc.get('image_count', 0)} images", unsafe_allow_html=False)
#                 if status == "failed" and doc.get("error_message"):
#                     st.caption(f"Error: {doc['error_message'][:150]}")
#     except Exception as e:
#         st.error(f"Couldn't load the index or document list: {e}")
#         st.caption(
#             "Run `python src/embed_and_index.py` at least once (with "
#             "OWNER_USER_ID set), and make sure GOOGLE_API_KEY, "
#             "SUPABASE_URL, and SUPABASE_ANON_KEY are all set."
#         )

#     st.divider()
#     st.subheader("Add a document")
#     uploaded = st.file_uploader("Drop a PDF here, or click to browse", type="pdf")
#     if uploaded and pipeline is not None and pg_client is not None and st.button("Process & add to index", type="primary", use_container_width=True):
#         with st.spinner(f"Extracting and indexing {uploaded.name} (captioning images takes a few minutes)..."):
#             try:
#                 n_added = process_uploaded_pdf(uploaded, pipeline, pg_client, st.session_state.auth_user_id)
#             except Exception as e:
#                 st.error(f"Processing failed: {e}")
#                 n_added = "error"
#         if n_added is None:
#             st.warning(f"'{uploaded.name}' already appears to be indexed. Skipped.")
#         elif n_added != "error":
#             st.success(f"Added {n_added} chunks from {uploaded.name}.")
#             st.rerun()

#     st.divider()
#     st.caption(
#         "Answers are grounded only in retrieved context (text, tables, and "
#         "chart images) from the documents above -- not the model's general "
#         "knowledge."
#     )

#     st.divider()
#     st.caption(f"Signed in as **{st.session_state.auth_email}**")
#     if st.button("Log out", use_container_width=True):
#         try:
#             sign_out(get_supabase_client(), st.session_state.auth_access_token, st.session_state.auth_refresh_token)
#         except Exception:
#             pass  # local session state is cleared regardless -- see auth_service.sign_out
#         for _key in ("auth_user_id", "auth_email", "auth_access_token", "auth_refresh_token"):
#             st.session_state[_key] = None
#         st.rerun()

# # ---------------- Main chat area ----------------
# st.markdown(
#     f'<div class="brand-header"><span style="font-size:1.7rem;">\U0001F9E0</span><h1>{APP_NAME}</h1></div>'
#     f'<div class="brand-tagline">{APP_TAGLINE}</div>',
#     unsafe_allow_html=True,
# )

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Set by a suggested-question button click below; consumed once, then
# # cleared, so it can't get reprocessed on an unrelated later rerun.
# pending_question = st.session_state.pop("pending_question", None)

# if not st.session_state.messages and not pending_question:
#     st.markdown(
#         '<div class="welcome-state">'
#         '<h1>Understand your documents with AI</h1>'
#         '<p>Upload a PDF in the sidebar and ask questions using grounded, '
#         'multimodal retrieval across text, tables, and charts.</p>'
#         '</div>',
#         unsafe_allow_html=True,
#     )
#     has_documents = pipeline is not None and len(user_documents) > 0
#     if has_documents:
#         st.markdown('<div class="suggested-label">Try asking</div>', unsafe_allow_html=True)
#         cols = st.columns(2)
#         for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
#             with cols[i % 2]:
#                 if st.button(suggestion, key=f"suggested_{i}", use_container_width=True):
#                     st.session_state.pending_question = suggestion
#                     st.rerun()
#     elif pipeline is not None:
#         st.info("No documents indexed yet -- add one from the sidebar to get started.")
# else:
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if message["role"] == "assistant" and message.get("sources"):
#                 render_sources(message["sources"])

# typed_question = st.chat_input("Ask a question about your documents...")
# question = pending_question or typed_question

# if pipeline is None:
#     st.info("Fix the index / API key issue in the sidebar before asking questions.")
# elif question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Retrieving and generating..."):
#             result = pipeline.answer(question, user_id=st.session_state.auth_user_id)
#         st.markdown(result["answer"])
#         render_sources(result["sources"])

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": result["answer"],
#         "sources": result["sources"],
#     })



# """
# DocuMind AI -- Streamlit UI

# General-purpose multimodal document intelligence: chat interface over the
# RAG pipeline (Postgres/pgvector for retrieval, Gemini for generation),
# plus a sidebar that lets you add new PDFs without leaving the app. Works
# over any structured PDF -- research papers, technical docs, manuals,
# reports -- not just financial filings. Uploading hands off to
# embed_and_index.process_and_index_pdf, the exact same function the CLI
# bulk-indexer uses, so there's exactly one place that logic lives.

# Run with:
#     streamlit run app.py
# """

# import html
# import sys
# from pathlib import Path

# import streamlit as st

# sys.path.insert(0, str(Path(__file__).parent / "src"))

# from rag_pipeline import RagPipeline  # noqa: E402
# from embed_and_index import process_and_index_pdf  # noqa: E402
# from auth_service import get_supabase_client, sign_up, sign_in, sign_out  # noqa: E402
# import db_service  # noqa: E402

# APP_NAME = "DocuMind AI"
# APP_TAGLINE = "Ask questions. Understand documents. Get grounded answers."
# SUGGESTED_QUESTIONS = [
#     "What are the key findings?",
#     "Summarize this document.",
#     "What does this chart show?",
#     "Compare the main metrics.",
# ]

# st.set_page_config(page_title=APP_NAME, page_icon="\U0001F9E0", layout="wide")

# # ---------------- Design system ----------------
# # Streamlit's own theme.toml drives native widgets (buttons, inputs, chat
# # bubbles); this CSS layer only styles the custom elements below it
# # (source cards, welcome state) that Streamlit has no built-in component
# # for. Deliberately uses soft/translucent borders and shadows rather than
# # hardcoded hex greys, so it degrades reasonably in both light and dark
# # mode instead of looking wrong in whichever one wasn't hand-tuned.
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

# html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

# .brand-header { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.1rem; }
# .brand-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
# .brand-tagline { color: rgba(120,120,130,0.9); font-size: 0.95rem; margin-bottom: 1.2rem; }

# .welcome-state { text-align: center; padding: 3rem 1rem 1.5rem 1rem; }
# .welcome-state h1 { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.5rem; }
# .welcome-state p { color: rgba(120,120,130,0.95); font-size: 1.05rem; max-width: 32rem; margin: 0 auto; }
# .suggested-label {
#     text-align: center; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em;
#     text-transform: uppercase; color: rgba(120,120,130,0.8); margin: 1.8rem 0 0.6rem 0;
# }

# .source-card {
#     border: 1px solid rgba(120,120,130,0.22);
#     border-radius: 10px;
#     padding: 0.55rem 0.85rem;
#     margin-bottom: 0.4rem;
#     box-shadow: 0 1px 2px rgba(0,0,0,0.04);
#     font-size: 0.88rem;
# }
# .source-card .doc-name { font-weight: 600; }
# .source-card .badge {
#     display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
#     text-transform: uppercase; padding: 0.05rem 0.5rem; border-radius: 999px;
#     background: rgba(79,70,229,0.12); color: #4F46E5; margin-left: 0.5rem;
# }
# </style>
# """, unsafe_allow_html=True)


# @st.cache_resource(show_spinner="Loading index and embedding model...")
# def load_pipeline() -> RagPipeline:
#     return RagPipeline()


# # ---------------- Authentication gate ----------------
# # Runs before anything else -- the embedding model below is only loaded
# # once someone is actually logged in.
# for _key in ("auth_user_id", "auth_email", "auth_access_token", "auth_refresh_token"):
#     if _key not in st.session_state:
#         st.session_state[_key] = None

# if not st.session_state.auth_access_token:
#     st.markdown(
#         f'<div class="brand-header"><span style="font-size:1.7rem;">\U0001F9E0</span><h1>{APP_NAME}</h1></div>'
#         f'<div class="brand-tagline">{APP_TAGLINE}</div>',
#         unsafe_allow_html=True,
#     )
#     try:
#         _auth_client = get_supabase_client()
#     except RuntimeError as e:
#         st.error(str(e))
#         st.stop()

#     login_tab, register_tab = st.tabs(["Log in", "Create account"])

#     with login_tab:
#         with st.form("login_form"):
#             login_email = st.text_input("Email", key="login_email")
#             login_password = st.text_input("Password", type="password", key="login_password")
#             login_submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
#         if login_submitted:
#             result = sign_in(_auth_client, login_email, login_password)
#             if result.success and result.access_token:
#                 st.session_state.auth_user_id = result.user_id
#                 st.session_state.auth_email = result.email
#                 st.session_state.auth_access_token = result.access_token
#                 st.session_state.auth_refresh_token = result.refresh_token
#                 st.rerun()
#             else:
#                 st.error(result.message)

#     with register_tab:
#         with st.form("register_form"):
#             reg_email = st.text_input("Email", key="register_email")
#             reg_password = st.text_input("Password", type="password", key="register_password",
#                                           help="At least 6 characters")
#             register_submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
#         if register_submitted:
#             result = sign_up(_auth_client, reg_email, reg_password)
#             if result.success and result.access_token:
#                 st.session_state.auth_user_id = result.user_id
#                 st.session_state.auth_email = result.email
#                 st.session_state.auth_access_token = result.access_token
#                 st.session_state.auth_refresh_token = result.refresh_token
#                 st.rerun()
#             elif result.success:
#                 st.info(result.message)  # created, but email confirmation is pending
#             else:
#                 st.error(result.message)

#     st.stop()


# def get_authenticated_client():
#     """Fresh Supabase client hydrated with the CURRENT user's session.
#     Deliberately NOT wrapped in @st.cache_resource -- that cache is
#     shared across every user of the app, so caching an authenticated
#     client would leak one user's session to everyone else. Constructing
#     a client is cheap (no network call happens until .execute()), so
#     there's no real cost to building it fresh each time it's needed."""
#     client = get_supabase_client()
#     client.auth.set_session(st.session_state.auth_access_token, st.session_state.auth_refresh_token)
#     return client


# def process_uploaded_pdf(uploaded_file, pipeline: RagPipeline, pg_client, user_id: str):
#     """Saves the upload to disk, then hands off to
#     embed_and_index.process_and_index_pdf -- the same function the CLI
#     bulk-indexer uses, so extraction/captioning/embedding/status-tracking
#     logic lives in exactly one place. Returns the number of chunks added,
#     or None if this filename is already indexed for this user. Raises on
#     genuine failure (already recorded in Postgres as "failed" by the time
#     it propagates here)."""
#     raw_dir = Path("data/raw_pdfs")
#     raw_dir.mkdir(parents=True, exist_ok=True)
#     pdf_path = raw_dir / uploaded_file.name
#     pdf_path.write_bytes(uploaded_file.getvalue())

#     result = process_and_index_pdf(pdf_path, pg_client, pipeline.embedder, user_id)
#     if result["status"] == "skipped":
#         return None
#     return result["chunks"]


# def render_sources(sources: list[dict]) -> None:
#     text_sources = [s for s in sources if s["content_type"] != "image"]
#     image_sources = [s for s in sources if s["content_type"] == "image"]

#     with st.expander(f"Sources ({len(sources)})"):
#         if text_sources:
#             seen = set()
#             for s in text_sources:
#                 key = (s["doc_id"], s["page_number"], s["content_type"])
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 doc_name = html.escape(str(s["doc_id"]))
#                 st.markdown(
#                     f'<div class="source-card">'
#                     f'<span class="doc-name">{doc_name}</span> &middot; page {s["page_number"]}'
#                     f'<span class="badge">{html.escape(s["content_type"])}</span>'
#                     f'</div>',
#                     unsafe_allow_html=True,
#                 )

#         if image_sources:
#             st.markdown("**Images used**")
#             seen_paths = set()
#             unique_images = []
#             for s in image_sources:
#                 path = s.get("image_path")
#                 if path and path not in seen_paths and Path(path).exists():
#                     seen_paths.add(path)
#                     unique_images.append(s)
#             cols = st.columns(min(len(unique_images), 3) or 1)
#             for i, s in enumerate(unique_images):
#                 with cols[i % len(cols)]:
#                     st.image(s["image_path"], caption=f"{s['doc_id']} p{s['page_number']}", use_container_width=True)


# # ---------------- Sidebar ----------------
# with st.sidebar:
#     st.markdown(f'<div class="brand-header"><span style="font-size:1.3rem;">\U0001F9E0</span>'
#                 f'<span style="font-weight:700;font-size:1.1rem;">{APP_NAME}</span></div>', unsafe_allow_html=True)
#     st.caption(APP_TAGLINE)
#     st.divider()

#     st.subheader("\U0001F4C1 Document Library")

#     pipeline = None
#     pg_client = None
#     user_documents = []
#     try:
#         pipeline = load_pipeline()
#         pg_client = get_authenticated_client()
#         user_documents = db_service.list_documents(pg_client, st.session_state.auth_user_id)

#         if not user_documents:
#             st.caption("No documents yet -- add one below.")
#         for doc in user_documents:
#             with st.container(border=True):
#                 st.markdown(f"**{html.escape(doc['filename'])}**")
#                 status = doc["status"]
#                 status_emoji = {"completed": "\u2705", "processing": "\u23F3", "failed": "\u274C", "uploading": "\u23F3"}.get(status, "\u2022")
#                 st.caption(f"{status_emoji} {status} &middot; {doc.get('chunk_count', 0)} chunks &middot; "
#                            f"{doc.get('image_count', 0)} images", unsafe_allow_html=False)
#                 if status == "failed" and doc.get("error_message"):
#                     st.caption(f"Error: {doc['error_message'][:150]}")
#     except Exception as e:
#         st.error(f"Couldn't load the index or document list: {e}")
#         st.caption(
#             "Make sure GOOGLE_API_KEY, SUPABASE_URL, and SUPABASE_ANON_KEY "
#             "are all set, and that supabase_schema.sql has been run in "
#             "your Supabase project's SQL Editor."
#         )

#     st.divider()
#     st.subheader("Add a document")
#     uploaded = st.file_uploader("Drop a PDF here, or click to browse", type="pdf")
#     if uploaded and pipeline is not None and pg_client is not None and st.button("Process & add to index", type="primary", use_container_width=True):
#         with st.spinner(f"Extracting and indexing {uploaded.name} (captioning images takes a few minutes)..."):
#             try:
#                 n_added = process_uploaded_pdf(uploaded, pipeline, pg_client, st.session_state.auth_user_id)
#             except Exception as e:
#                 st.error(f"Processing failed: {e}")
#                 n_added = "error"
#         if n_added is None:
#             st.warning(f"'{uploaded.name}' already appears to be indexed. Skipped.")
#         elif n_added != "error":
#             st.success(f"Added {n_added} chunks from {uploaded.name}.")
#             st.rerun()

#     st.divider()
#     st.caption(
#         "Answers are grounded only in retrieved context (text, tables, and "
#         "chart images) from the documents above -- not the model's general "
#         "knowledge."
#     )

#     st.divider()
#     st.caption(f"Signed in as **{st.session_state.auth_email}**")
#     if st.button("Log out", use_container_width=True):
#         try:
#             sign_out(get_supabase_client(), st.session_state.auth_access_token, st.session_state.auth_refresh_token)
#         except Exception:
#             pass  # local session state is cleared regardless -- see auth_service.sign_out
#         for _key in ("auth_user_id", "auth_email", "auth_access_token", "auth_refresh_token"):
#             st.session_state[_key] = None
#         st.rerun()

# # ---------------- Main chat area ----------------
# st.markdown(
#     f'<div class="brand-header"><span style="font-size:1.7rem;">\U0001F9E0</span><h1>{APP_NAME}</h1></div>'
#     f'<div class="brand-tagline">{APP_TAGLINE}</div>',
#     unsafe_allow_html=True,
# )

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Set by a suggested-question button click below; consumed once, then
# # cleared, so it can't get reprocessed on an unrelated later rerun.
# pending_question = st.session_state.pop("pending_question", None)

# if not st.session_state.messages and not pending_question:
#     st.markdown(
#         '<div class="welcome-state">'
#         '<h1>Understand your documents with AI</h1>'
#         '<p>Upload a PDF in the sidebar and ask questions using grounded, '
#         'multimodal retrieval across text, tables, and charts.</p>'
#         '</div>',
#         unsafe_allow_html=True,
#     )
#     has_documents = pipeline is not None and len(user_documents) > 0
#     if has_documents:
#         st.markdown('<div class="suggested-label">Try asking</div>', unsafe_allow_html=True)
#         cols = st.columns(2)
#         for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
#             with cols[i % 2]:
#                 if st.button(suggestion, key=f"suggested_{i}", use_container_width=True):
#                     st.session_state.pending_question = suggestion
#                     st.rerun()
#     elif pipeline is not None:
#         st.info("No documents indexed yet -- add one from the sidebar to get started.")
# else:
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if message["role"] == "assistant" and message.get("sources"):
#                 render_sources(message["sources"])

# typed_question = st.chat_input("Ask a question about your documents...")
# question = pending_question or typed_question

# if pipeline is None or pg_client is None:
#     st.info("Fix the index / API key issue in the sidebar before asking questions.")
# elif question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Retrieving and generating..."):
#             result = pipeline.answer(question, pg_client, st.session_state.auth_user_id)
#         st.markdown(result["answer"])
#         render_sources(result["sources"])

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": result["answer"],
#         "sources": result["sources"],
#     })


# """
# DocuMind AI -- Streamlit UI

# General-purpose multimodal document intelligence: chat interface over the
# RAG pipeline (Postgres/pgvector for retrieval, Gemini for generation),
# plus a sidebar that lets you add new PDFs without leaving the app. Works
# over any structured PDF -- research papers, technical docs, manuals,
# reports -- not just financial filings. Uploading hands off to
# embed_and_index.process_and_index_pdf, the exact same function the CLI
# bulk-indexer uses, so there's exactly one place that logic lives.

# Run with:
#     streamlit run app.py
# """

# import html
# import sys
# from pathlib import Path

# try:
#     from dotenv import load_dotenv
#     load_dotenv()  # local dev only -- Streamlit Cloud secrets sync to
#                      # os.environ on their own, this is a no-op there
# except ImportError:
#     pass

# import streamlit as st

# sys.path.insert(0, str(Path(__file__).parent / "src"))

# from rag_pipeline import RagPipeline  # noqa: E402
# from embed_and_index import process_and_index_pdf  # noqa: E402
# from auth_service import get_supabase_client, sign_up, sign_in, sign_out  # noqa: E402
# import db_service  # noqa: E402

# APP_NAME = "DocuMind AI"
# APP_TAGLINE = "Ask questions. Understand documents. Get grounded answers."
# SUGGESTED_QUESTIONS = [
#     "What are the key findings?",
#     "Summarize this document.",
#     "What does this chart show?",
#     "Compare the main metrics.",
# ]

# st.set_page_config(page_title=APP_NAME, page_icon="\U0001F9E0", layout="wide")

# # ---------------- Design system ----------------
# # Streamlit's own theme.toml drives native widgets (buttons, inputs, chat
# # bubbles); this CSS layer only styles the custom elements below it
# # (source cards, welcome state) that Streamlit has no built-in component
# # for. Deliberately uses soft/translucent borders and shadows rather than
# # hardcoded hex greys, so it degrades reasonably in both light and dark
# # mode instead of looking wrong in whichever one wasn't hand-tuned.
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

# html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

# .brand-header { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.1rem; }
# .brand-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
# .brand-tagline { color: rgba(120,120,130,0.9); font-size: 0.95rem; margin-bottom: 1.2rem; }

# .welcome-state { text-align: center; padding: 3rem 1rem 1.5rem 1rem; }
# .welcome-state h1 { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.5rem; }
# .welcome-state p { color: rgba(120,120,130,0.95); font-size: 1.05rem; max-width: 32rem; margin: 0 auto; }
# .suggested-label {
#     text-align: center; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em;
#     text-transform: uppercase; color: rgba(120,120,130,0.8); margin: 1.8rem 0 0.6rem 0;
# }

# .source-card {
#     border: 1px solid rgba(120,120,130,0.22);
#     border-radius: 10px;
#     padding: 0.55rem 0.85rem;
#     margin-bottom: 0.4rem;
#     box-shadow: 0 1px 2px rgba(0,0,0,0.04);
#     font-size: 0.88rem;
# }
# .source-card .doc-name { font-weight: 600; }
# .source-card .badge {
#     display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
#     text-transform: uppercase; padding: 0.05rem 0.5rem; border-radius: 999px;
#     background: rgba(79,70,229,0.12); color: #4F46E5; margin-left: 0.5rem;
# }
# </style>
# """, unsafe_allow_html=True)


# @st.cache_resource(show_spinner="Loading index and embedding model...")
# def load_pipeline() -> RagPipeline:
#     return RagPipeline()


# # ---------------- Authentication gate ----------------
# # Runs before anything else -- the embedding model below is only loaded
# # once someone is actually logged in.
# for _key in ("auth_user_id", "auth_email", "auth_access_token", "auth_refresh_token"):
#     if _key not in st.session_state:
#         st.session_state[_key] = None

# if not st.session_state.auth_access_token:
#     st.markdown(
#         f'<div class="brand-header"><span style="font-size:1.7rem;">\U0001F9E0</span><h1>{APP_NAME}</h1></div>'
#         f'<div class="brand-tagline">{APP_TAGLINE}</div>',
#         unsafe_allow_html=True,
#     )
#     try:
#         _auth_client = get_supabase_client()
#     except RuntimeError as e:
#         st.error(str(e))
#         st.stop()

#     login_tab, register_tab = st.tabs(["Log in", "Create account"])

#     with login_tab:
#         with st.form("login_form"):
#             login_email = st.text_input("Email", key="login_email")
#             login_password = st.text_input("Password", type="password", key="login_password")
#             login_submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
#         if login_submitted:
#             result = sign_in(_auth_client, login_email, login_password)
#             if result.success and result.access_token:
#                 st.session_state.auth_user_id = result.user_id
#                 st.session_state.auth_email = result.email
#                 st.session_state.auth_access_token = result.access_token
#                 st.session_state.auth_refresh_token = result.refresh_token
#                 st.rerun()
#             else:
#                 st.error(result.message)

#     with register_tab:
#         with st.form("register_form"):
#             reg_email = st.text_input("Email", key="register_email")
#             reg_password = st.text_input("Password", type="password", key="register_password",
#                                           help="At least 6 characters")
#             register_submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
#         if register_submitted:
#             result = sign_up(_auth_client, reg_email, reg_password)
#             if result.success and result.access_token:
#                 st.session_state.auth_user_id = result.user_id
#                 st.session_state.auth_email = result.email
#                 st.session_state.auth_access_token = result.access_token
#                 st.session_state.auth_refresh_token = result.refresh_token
#                 st.rerun()
#             elif result.success:
#                 st.info(result.message)  # created, but email confirmation is pending
#             else:
#                 st.error(result.message)

#     st.stop()


# def get_authenticated_client():
#     """Fresh Supabase client hydrated with the CURRENT user's session.
#     Deliberately NOT wrapped in @st.cache_resource -- that cache is
#     shared across every user of the app, so caching an authenticated
#     client would leak one user's session to everyone else. Constructing
#     a client is cheap (no network call happens until .execute()), so
#     there's no real cost to building it fresh each time it's needed."""
#     client = get_supabase_client()
#     client.auth.set_session(st.session_state.auth_access_token, st.session_state.auth_refresh_token)
#     return client


# def process_uploaded_pdf(uploaded_file, pipeline: RagPipeline, pg_client, user_id: str):
#     """Saves the upload to disk, then hands off to
#     embed_and_index.process_and_index_pdf -- the same function the CLI
#     bulk-indexer uses, so extraction/captioning/embedding/status-tracking
#     logic lives in exactly one place. Returns the number of chunks added,
#     or None if this filename is already indexed for this user. Raises on
#     genuine failure (already recorded in Postgres as "failed" by the time
#     it propagates here)."""
#     raw_dir = Path("data/raw_pdfs")
#     raw_dir.mkdir(parents=True, exist_ok=True)
#     pdf_path = raw_dir / uploaded_file.name
#     pdf_path.write_bytes(uploaded_file.getvalue())

#     result = process_and_index_pdf(pdf_path, pg_client, pipeline.embedder, user_id)
#     if result["status"] == "skipped":
#         return None
#     return result["chunks"]


# def render_sources(sources: list[dict]) -> None:
#     text_sources = [s for s in sources if s["content_type"] != "image"]
#     image_sources = [s for s in sources if s["content_type"] == "image"]

#     with st.expander(f"Sources ({len(sources)})"):
#         if text_sources:
#             seen = set()
#             for s in text_sources:
#                 key = (s["doc_id"], s["page_number"], s["content_type"])
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 doc_name = html.escape(str(s["doc_id"]))
#                 st.markdown(
#                     f'<div class="source-card">'
#                     f'<span class="doc-name">{doc_name}</span> &middot; page {s["page_number"]}'
#                     f'<span class="badge">{html.escape(s["content_type"])}</span>'
#                     f'</div>',
#                     unsafe_allow_html=True,
#                 )

#         if image_sources:
#             st.markdown("**Images used**")
#             seen_paths = set()
#             unique_images = []
#             for s in image_sources:
#                 path = s.get("image_path")
#                 if path and path not in seen_paths and Path(path).exists():
#                     seen_paths.add(path)
#                     unique_images.append(s)
#             cols = st.columns(min(len(unique_images), 3) or 1)
#             for i, s in enumerate(unique_images):
#                 with cols[i % len(cols)]:
#                     st.image(s["image_path"], caption=f"{s['doc_id']} p{s['page_number']}", use_container_width=True)


# # ---------------- Sidebar ----------------
# with st.sidebar:
#     st.markdown(f'<div class="brand-header"><span style="font-size:1.3rem;">\U0001F9E0</span>'
#                 f'<span style="font-weight:700;font-size:1.1rem;">{APP_NAME}</span></div>', unsafe_allow_html=True)
#     st.caption(APP_TAGLINE)
#     st.divider()

#     st.subheader("\U0001F4C1 Document Library")

#     pipeline = None
#     pg_client = None
#     user_documents = []
#     try:
#         pipeline = load_pipeline()
#         pg_client = get_authenticated_client()
#         user_documents = db_service.list_documents(pg_client, st.session_state.auth_user_id)

#         if not user_documents:
#             st.caption("No documents yet -- add one below.")
#         for doc in user_documents:
#             with st.container(border=True):
#                 st.markdown(f"**{html.escape(doc['filename'])}**")
#                 status = doc["status"]
#                 status_emoji = {"completed": "\u2705", "processing": "\u23F3", "failed": "\u274C", "uploading": "\u23F3"}.get(status, "\u2022")
#                 st.caption(f"{status_emoji} {status} &middot; {doc.get('chunk_count', 0)} chunks &middot; "
#                            f"{doc.get('image_count', 0)} images", unsafe_allow_html=False)
#                 if status == "failed" and doc.get("error_message"):
#                     st.caption(f"Error: {doc['error_message'][:150]}")
#     except Exception as e:
#         st.error(f"Couldn't load the index or document list: {e}")
#         st.caption(
#             "Make sure GOOGLE_API_KEY, SUPABASE_URL, and SUPABASE_ANON_KEY "
#             "are all set, and that supabase_schema.sql has been run in "
#             "your Supabase project's SQL Editor."
#         )

#     st.divider()
#     st.subheader("Add a document")
#     uploaded = st.file_uploader("Drop a PDF here, or click to browse", type="pdf")
#     if uploaded and pipeline is not None and pg_client is not None and st.button("Process & add to index", type="primary", use_container_width=True):
#         with st.spinner(f"Extracting and indexing {uploaded.name} (captioning images takes a few minutes)..."):
#             try:
#                 n_added = process_uploaded_pdf(uploaded, pipeline, pg_client, st.session_state.auth_user_id)
#             except Exception as e:
#                 st.error(f"Processing failed: {e}")
#                 n_added = "error"
#         if n_added is None:
#             st.warning(f"'{uploaded.name}' already appears to be indexed. Skipped.")
#         elif n_added != "error":
#             st.success(f"Added {n_added} chunks from {uploaded.name}.")
#             st.rerun()

#     st.divider()
#     st.caption(
#         "Answers are grounded only in retrieved context (text, tables, and "
#         "chart images) from the documents above -- not the model's general "
#         "knowledge."
#     )

#     st.divider()
#     st.caption(f"Signed in as **{st.session_state.auth_email}**")
#     if st.button("Log out", use_container_width=True):
#         try:
#             sign_out(get_supabase_client(), st.session_state.auth_access_token, st.session_state.auth_refresh_token)
#         except Exception:
#             pass  # local session state is cleared regardless -- see auth_service.sign_out
#         for _key in ("auth_user_id", "auth_email", "auth_access_token", "auth_refresh_token"):
#             st.session_state[_key] = None
#         st.rerun()

# # ---------------- Main chat area ----------------
# st.markdown(
#     f'<div class="brand-header"><span style="font-size:1.7rem;">\U0001F9E0</span><h1>{APP_NAME}</h1></div>'
#     f'<div class="brand-tagline">{APP_TAGLINE}</div>',
#     unsafe_allow_html=True,
# )

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Set by a suggested-question button click below; consumed once, then
# # cleared, so it can't get reprocessed on an unrelated later rerun.
# pending_question = st.session_state.pop("pending_question", None)

# if not st.session_state.messages and not pending_question:
#     st.markdown(
#         '<div class="welcome-state">'
#         '<h1>Understand your documents with AI</h1>'
#         '<p>Upload a PDF in the sidebar and ask questions using grounded, '
#         'multimodal retrieval across text, tables, and charts.</p>'
#         '</div>',
#         unsafe_allow_html=True,
#     )
#     has_documents = pipeline is not None and len(user_documents) > 0
#     if has_documents:
#         st.markdown('<div class="suggested-label">Try asking</div>', unsafe_allow_html=True)
#         cols = st.columns(2)
#         for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
#             with cols[i % 2]:
#                 if st.button(suggestion, key=f"suggested_{i}", use_container_width=True):
#                     st.session_state.pending_question = suggestion
#                     st.rerun()
#     elif pipeline is not None:
#         st.info("No documents indexed yet -- add one from the sidebar to get started.")
# else:
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if message["role"] == "assistant" and message.get("sources"):
#                 render_sources(message["sources"])

# typed_question = st.chat_input("Ask a question about your documents...")
# question = pending_question or typed_question

# if pipeline is None or pg_client is None:
#     st.info("Fix the index / API key issue in the sidebar before asking questions.")
# elif question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Retrieving and generating..."):
#             result = pipeline.answer(question, pg_client, st.session_state.auth_user_id)
#         st.markdown(result["answer"])
#         render_sources(result["sources"])

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": result["answer"],
#         "sources": result["sources"],
#     })





"""
DocuMind AI -- Streamlit UI

General-purpose multimodal document intelligence: chat interface over the
RAG pipeline (Postgres/pgvector for retrieval, Gemini for generation),
plus a sidebar that lets you add new PDFs without leaving the app. Works
over any structured PDF -- research papers, technical docs, manuals,
reports -- not just financial filings. Uploading hands off to
embed_and_index.process_and_index_pdf, the exact same function the CLI
bulk-indexer uses, so there's exactly one place that logic lives.

Run with:
    streamlit run app.py
"""

import html
import io
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # local dev only -- Streamlit Cloud secrets sync to
                     # os.environ on their own, this is a no-op there
except ImportError:
    pass

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import RagPipeline  # noqa: E402
from embed_and_index import process_and_index_pdf  # noqa: E402
from auth_service import get_supabase_client, sign_up, sign_in, sign_out  # noqa: E402
import db_service  # noqa: E402
import storage_service  # noqa: E402

APP_NAME = "DocuMind AI"
APP_TAGLINE = "Ask questions. Understand documents. Get grounded answers."
SUGGESTED_QUESTIONS = [
    "What are the key findings?",
    "Summarize this document.",
    "What does this chart show?",
    "Compare the main metrics.",
]

st.set_page_config(page_title=APP_NAME, page_icon="\U0001F9E0", layout="wide")

# ---------------- Design system ----------------
# Streamlit's own theme.toml drives native widgets (buttons, inputs, chat
# bubbles); this CSS layer only styles the custom elements below it
# (source cards, welcome state) that Streamlit has no built-in component
# for. Deliberately uses soft/translucent borders and shadows rather than
# hardcoded hex greys, so it degrades reasonably in both light and dark
# mode instead of looking wrong in whichever one wasn't hand-tuned.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.brand-header { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.1rem; }
.brand-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
.brand-tagline { color: rgba(120,120,130,0.9); font-size: 0.95rem; margin-bottom: 1.2rem; }

.welcome-state { text-align: center; padding: 3rem 1rem 1.5rem 1rem; }
.welcome-state h1 { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.5rem; }
.welcome-state p { color: rgba(120,120,130,0.95); font-size: 1.05rem; max-width: 32rem; margin: 0 auto; }
.suggested-label {
    text-align: center; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase; color: rgba(120,120,130,0.8); margin: 1.8rem 0 0.6rem 0;
}

.source-card {
    border: 1px solid rgba(120,120,130,0.22);
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.4rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    font-size: 0.88rem;
}
.source-card .doc-name { font-weight: 600; }
.source-card .badge {
    display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
    text-transform: uppercase; padding: 0.05rem 0.5rem; border-radius: 999px;
    background: rgba(79,70,229,0.12); color: #4F46E5; margin-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_pipeline() -> RagPipeline:
    return RagPipeline()


# ---------------- Authentication gate ----------------
# Runs before anything else -- the embedding model below is only loaded
# once someone is actually logged in.
for _key in ("auth_user_id", "auth_email", "auth_access_token", "auth_refresh_token"):
    if _key not in st.session_state:
        st.session_state[_key] = None

if not st.session_state.auth_access_token:
    st.markdown(
        f'<div class="brand-header"><span style="font-size:1.7rem;">\U0001F9E0</span><h1>{APP_NAME}</h1></div>'
        f'<div class="brand-tagline">{APP_TAGLINE}</div>',
        unsafe_allow_html=True,
    )
    try:
        _auth_client = get_supabase_client()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if login_submitted:
            result = sign_in(_auth_client, login_email, login_password)
            if result.success and result.access_token:
                st.session_state.auth_user_id = result.user_id
                st.session_state.auth_email = result.email
                st.session_state.auth_access_token = result.access_token
                st.session_state.auth_refresh_token = result.refresh_token
                st.rerun()
            else:
                st.error(result.message)

    with register_tab:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="register_email")
            reg_password = st.text_input("Password", type="password", key="register_password",
                                          help="At least 6 characters")
            register_submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if register_submitted:
            result = sign_up(_auth_client, reg_email, reg_password)
            if result.success and result.access_token:
                st.session_state.auth_user_id = result.user_id
                st.session_state.auth_email = result.email
                st.session_state.auth_access_token = result.access_token
                st.session_state.auth_refresh_token = result.refresh_token
                st.rerun()
            elif result.success:
                st.info(result.message)  # created, but email confirmation is pending
            else:
                st.error(result.message)

    st.stop()


def get_authenticated_client():
    """Fresh Supabase client hydrated with the CURRENT user's session.
    Deliberately NOT wrapped in @st.cache_resource -- that cache is
    shared across every user of the app, so caching an authenticated
    client would leak one user's session to everyone else. Constructing
    a client is cheap (no network call happens until .execute()), so
    there's no real cost to building it fresh each time it's needed."""
    client = get_supabase_client()
    client.auth.set_session(st.session_state.auth_access_token, st.session_state.auth_refresh_token)
    return client


def process_uploaded_pdf(uploaded_file, pipeline: RagPipeline, pg_client, user_id: str):
    """Saves the upload to disk, then hands off to
    embed_and_index.process_and_index_pdf -- the same function the CLI
    bulk-indexer uses, so extraction/captioning/embedding/status-tracking
    logic lives in exactly one place. Returns the number of chunks added,
    or None if this filename is already indexed for this user. Raises on
    genuine failure (already recorded in Postgres as "failed" by the time
    it propagates here)."""
    raw_dir = Path("data/raw_pdfs")
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getvalue())

    result = process_and_index_pdf(pdf_path, pg_client, pipeline.embedder, user_id)
    if result["status"] == "skipped":
        return None
    return result["chunks"]


@st.cache_data(show_spinner=False)
def _download_source_image(_pg_client, storage_path: str):
    """Returns a decoded PIL Image, or None if the download/decode
    failed. Cached so re-rendering old chat history on every Streamlit
    rerun doesn't re-download the same image from Storage every time --
    only storage_path determines cache hits. _pg_client is prefixed with
    an underscore specifically so Streamlit excludes it from the cache
    key/hashing (it isn't a hashable object, and it shouldn't affect
    which cache entry is used anyway -- the image content only depends
    on the path, not which client fetched it)."""
    try:
        file_bytes = storage_service.download_bytes(_pg_client, storage_path)
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return None


def render_sources(sources: list[dict], pg_client) -> None:
    text_sources = [s for s in sources if s["content_type"] != "image"]
    image_sources = [s for s in sources if s["content_type"] == "image"]

    with st.expander(f"Sources ({len(sources)})"):
        if text_sources:
            seen = set()
            for s in text_sources:
                key = (s["doc_id"], s["page_number"], s["content_type"])
                if key in seen:
                    continue
                seen.add(key)
                doc_name = html.escape(str(s["doc_id"]))
                st.markdown(
                    f'<div class="source-card">'
                    f'<span class="doc-name">{doc_name}</span> &middot; page {s["page_number"]}'
                    f'<span class="badge">{html.escape(s["content_type"])}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if image_sources:
            seen_paths = set()
            unique_sources = []
            for s in image_sources:
                path = s.get("image_path")
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    unique_sources.append(s)

            # Download+decode before printing the header, so "Images
            # used" only appears when something actually loaded --
            # matches the old behavior of silently skipping missing ones.
            loaded = [(s, _download_source_image(pg_client, s["image_path"])) for s in unique_sources]
            loaded = [(s, img) for s, img in loaded if img is not None]

            if loaded:
                st.markdown("**Images used**")
                cols = st.columns(min(len(loaded), 3) or 1)
                for i, (s, img) in enumerate(loaded):
                    with cols[i % len(cols)]:
                        st.image(img, caption=f"{s['doc_id']} p{s['page_number']}", use_container_width=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(f'<div class="brand-header"><span style="font-size:1.3rem;">\U0001F9E0</span>'
                f'<span style="font-weight:700;font-size:1.1rem;">{APP_NAME}</span></div>', unsafe_allow_html=True)
    st.caption(APP_TAGLINE)
    st.divider()

    st.subheader("\U0001F4C1 Document Library")

    pipeline = None
    pg_client = None
    user_documents = []
    try:
        pipeline = load_pipeline()
        pg_client = get_authenticated_client()
        user_documents = db_service.list_documents(pg_client, st.session_state.auth_user_id)

        if not user_documents:
            st.caption("No documents yet -- add one below.")
        for doc in user_documents:
            with st.container(border=True):
                st.markdown(f"**{html.escape(doc['filename'])}**")
                status = doc["status"]
                status_emoji = {"completed": "\u2705", "processing": "\u23F3", "failed": "\u274C", "uploading": "\u23F3"}.get(status, "\u2022")
                st.caption(f"{status_emoji} {status} &middot; {doc.get('chunk_count', 0)} chunks &middot; "
                           f"{doc.get('image_count', 0)} images", unsafe_allow_html=False)
                if status == "failed" and doc.get("error_message"):
                    st.caption(f"Error: {doc['error_message'][:150]}")
    except Exception as e:
        st.error(f"Couldn't load the index or document list: {e}")
        st.caption(
            "Make sure GOOGLE_API_KEY, SUPABASE_URL, and SUPABASE_ANON_KEY "
            "are all set, and that supabase_schema.sql has been run in "
            "your Supabase project's SQL Editor."
        )

    st.divider()
    st.subheader("Add a document")
    uploaded = st.file_uploader("Drop a PDF here, or click to browse", type="pdf")
    if uploaded and pipeline is not None and pg_client is not None and st.button("Process & add to index", type="primary", use_container_width=True):
        with st.spinner(f"Extracting and indexing {uploaded.name} (captioning images takes a few minutes)..."):
            try:
                n_added = process_uploaded_pdf(uploaded, pipeline, pg_client, st.session_state.auth_user_id)
            except Exception as e:
                st.error(f"Processing failed: {e}")
                n_added = "error"
        if n_added is None:
            st.warning(f"'{uploaded.name}' already appears to be indexed. Skipped.")
        elif n_added != "error":
            st.success(f"Added {n_added} chunks from {uploaded.name}.")
            st.rerun()

    st.divider()
    st.caption(
        "Answers are grounded only in retrieved context (text, tables, and "
        "chart images) from the documents above -- not the model's general "
        "knowledge."
    )

    st.divider()
    st.caption(f"Signed in as **{st.session_state.auth_email}**")
    if st.button("Log out", use_container_width=True):
        try:
            sign_out(get_supabase_client(), st.session_state.auth_access_token, st.session_state.auth_refresh_token)
        except Exception:
            pass  # local session state is cleared regardless -- see auth_service.sign_out
        for _key in ("auth_user_id", "auth_email", "auth_access_token", "auth_refresh_token"):
            st.session_state[_key] = None
        st.rerun()

# ---------------- Main chat area ----------------
st.markdown(
    f'<div class="brand-header"><span style="font-size:1.7rem;">\U0001F9E0</span><h1>{APP_NAME}</h1></div>'
    f'<div class="brand-tagline">{APP_TAGLINE}</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Set by a suggested-question button click below; consumed once, then
# cleared, so it can't get reprocessed on an unrelated later rerun.
pending_question = st.session_state.pop("pending_question", None)

if not st.session_state.messages and not pending_question:
    st.markdown(
        '<div class="welcome-state">'
        '<h1>Understand your documents with AI</h1>'
        '<p>Upload a PDF in the sidebar and ask questions using grounded, '
        'multimodal retrieval across text, tables, and charts.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    has_documents = pipeline is not None and len(user_documents) > 0
    if has_documents:
        st.markdown('<div class="suggested-label">Try asking</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggested_{i}", use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()
    elif pipeline is not None:
        st.info("No documents indexed yet -- add one from the sidebar to get started.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"], pg_client)

typed_question = st.chat_input("Ask a question about your documents...")
question = pending_question or typed_question

if pipeline is None or pg_client is None:
    st.info("Fix the index / API key issue in the sidebar before asking questions.")
elif question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = pipeline.answer(question, pg_client, st.session_state.auth_user_id)
        st.markdown(result["answer"])
        render_sources(result["sources"], pg_client)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })