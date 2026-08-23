"""
Supabase Storage access for raw PDFs and extracted images/page renders --
this is what makes the app's multimodal grounding survive a redeploy.
Before this, extracted images lived only on local disk: retrieval would
still "find" an image chunk after a redeploy wiped that disk (the caption
lives in Postgres), but load_retrieved_images() would silently skip the
missing file -- the app wouldn't crash, it would just quietly stop
showing the model any actual pictures. No error, just degraded answers.

Path scheme: {user_id}/{document_id}/original.pdf for the raw PDF, and
{user_id}/{document_id}/images/{filename} for extracted images. This is
what the Storage RLS policy in supabase_schema.sql checks (first path
segment must equal auth.uid()) -- verified locally against a real
Postgres RLS policy using this exact scheme, including that a user
genuinely cannot write into another user's folder. It's also what makes
two different users uploading a same-named file safe, closing the
local-disk collision gap documented in ingest.py.

Same critical note as db_service.py: every function here needs a client
that's already had auth.set_session() called on it. RLS on
storage.objects means an unauthenticated client just sees/writes nothing
for this private bucket, not an error.
"""

from pathlib import Path

from supabase import Client

BUCKET_NAME = "documents"


def pdf_storage_path(user_id: str, document_id: str) -> str:
    return f"{user_id}/{document_id}/original.pdf"


def image_storage_path(user_id: str, document_id: str, image_filename: str) -> str:
    return f"{user_id}/{document_id}/images/{image_filename}"


def upload_pdf(client: Client, user_id: str, document_id: str, file_bytes: bytes) -> str:
    path = pdf_storage_path(user_id, document_id)
    client.storage.from_(BUCKET_NAME).upload(
        path, file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    return path


def upload_image(client: Client, user_id: str, document_id: str, local_path: Path) -> str:
    storage_path = image_storage_path(user_id, document_id, local_path.name)
    content_type = "image/png" if local_path.suffix.lower() == ".png" else "image/jpeg"
    client.storage.from_(BUCKET_NAME).upload(
        storage_path, local_path.read_bytes(),
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return storage_path


def download_bytes(client: Client, storage_path: str) -> bytes:
    return client.storage.from_(BUCKET_NAME).download(storage_path)


def delete_document_files(client: Client, user_id: str, document_id: str) -> None:
    """Best-effort cleanup -- removes everything under this document's
    storage folder. Postgres cascade already handles the documents/chunks
    rows automatically when a document is deleted; storage needs this
    explicit call since it isn't a Postgres foreign key and would
    otherwise accumulate orphaned files. NOTE: less rigorously tested
    than upload/download -- list()'s exact return shape for folder vs.
    file entries couldn't be verified against a live bucket from this
    environment. Wrapped defensively; a failure here should never block
    the (more important) document-row deletion that calls it."""
    prefix = f"{user_id}/{document_id}"
    all_paths = []
    for sub in ("", "/images"):
        try:
            listing = client.storage.from_(BUCKET_NAME).list(f"{prefix}{sub}")
        except Exception:
            continue
        for item in listing:
            if item.get("id") is not None:  # skip folder placeholder entries
                all_paths.append(f"{prefix}{sub}/{item['name']}")
    if all_paths:
        try:
            client.storage.from_(BUCKET_NAME).remove(all_paths)
        except Exception:
            pass
