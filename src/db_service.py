"""
Postgres access for document metadata (relational layer -- embeddings
still live entirely in ChromaDB; see rag_pipeline.py / embed_and_index.py
for that half).

CRITICAL: every function here expects a client that's already had
auth.set_session() called on it with the current user's tokens. Row
Level Security on the `documents` table checks auth.uid() -- a client
still running on the bare anon key (no session set) isn't an error, it
just sees zero rows for everything, since RLS treats it as no
authenticated user. If a query here ever comes back unexpectedly empty,
check that the session was actually set before this was called, before
assuming the query itself is wrong.

conversations/messages tables already exist in supabase_schema.sql but
aren't wired up here yet -- that's the next step, kept separate rather
than growing this file ahead of what's actually being used.
"""

from typing import Optional

from supabase import Client


def create_document(client: Client, user_id: str, filename: str, file_size_bytes: int) -> dict:
    response = client.table("documents").insert({
        "user_id": user_id,
        "filename": filename,
        "file_size_bytes": file_size_bytes,
        "status": "uploading",
    }).execute()
    return response.data[0]


def update_document(client: Client, document_id: str, **fields) -> None:
    """e.g. update_document(client, doc_id, status="completed", chunk_count=42, image_count=7)"""
    client.table("documents").update(fields).eq("id", document_id).execute()


def list_documents(client: Client, user_id: str) -> list[dict]:
    response = (
        client.table("documents")
        .select("*")
        .eq("user_id", user_id)  # redundant with RLS, kept explicit and self-documenting on purpose
        .order("created_at", desc=True)
        .execute()
    )
    return response.data


def get_document_by_filename(client: Client, user_id: str, filename: str) -> Optional[dict]:
    response = (
        client.table("documents")
        .select("*")
        .eq("user_id", user_id)
        .eq("filename", filename)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def delete_document(client: Client, document_id: str) -> None:
    client.table("documents").delete().eq("id", document_id).execute()
