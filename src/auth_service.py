"""
Supabase-backed authentication.

Uses Supabase's built-in Auth (GoTrue) instead of hand-rolled password
hashing/session code -- one less thing to get wrong, and the actual
reason Supabase was picked over a bare self-hosted Postgres in the first
place: managed auth AND a persistent database from one free project.

------------------------------------------------------------------------
ONE-TIME SETUP (in the Supabase dashboard, supabase.com):
------------------------------------------------------------------------
1. Create a free project.
2. Project Settings -> API -> copy the "Project URL" and the "anon
   public" key. Do NOT copy the "service_role" key for this -- that key
   bypasses Row Level Security entirely and must never be used in code
   that runs on behalf of end users.
3. Authentication -> Providers -> Email -> for a personal / small-
   audience app, consider turning OFF "Confirm email". With it on,
   sign-up returns a user but no session until they click an emailed
   confirmation link -- and Streamlit has no clean way to receive that
   redirect. Turn it back on if this ever becomes a genuinely public app.
4. Copy .env.example to .env and fill in SUPABASE_URL and
   SUPABASE_ANON_KEY (same pattern as your existing GOOGLE_API_KEY).

The anon key is SAFE to use here -- it's the public client key Supabase
is designed to have embedded in frontend code. Protection comes from Row
Level Security policies on the data itself (added in a later step, once
the documents/conversations tables exist), not from keeping this key
secret. This is different from the service_role key, which must stay
server-only and isn't used anywhere in this module.
------------------------------------------------------------------------
"""

import os
from dataclasses import dataclass
from typing import Optional

from supabase import create_client, Client
from supabase_auth.errors import AuthApiError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fine in production, where real env vars are set directly


@dataclass
class AuthResult:
    success: bool
    message: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None


def _get_config_value(key: str) -> Optional[str]:
    """Prefer real environment variables (works everywhere, matches how
    GOOGLE_API_KEY is already read); fall back to Streamlit secrets so
    this also works unmodified once deployed to Streamlit Community
    Cloud, which surfaces secrets.toml differently than a plain .env."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


def get_supabase_client() -> Client:
    url = _get_config_value("SUPABASE_URL")
    key = _get_config_value("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_ANON_KEY before running this -- "
            "see the setup steps at the top of src/auth_service.py."
        )
    return create_client(url, key)


def sign_up(client: Client, email: str, password: str) -> AuthResult:
    try:
        response = client.auth.sign_up({"email": email, "password": password})
    except AuthApiError as e:
        return AuthResult(success=False, message=e.message)

    if response.user is None:
        return AuthResult(success=False, message="Sign-up failed for an unknown reason.")

    if response.session is None:
        # Email confirmation is still ON in the Supabase dashboard -- see
        # the setup note above if you want signup to log the user in
        # immediately instead.
        return AuthResult(
            success=True,
            message="Account created. Check your email to confirm it, then log in.",
            user_id=response.user.id,
            email=response.user.email,
        )

    return AuthResult(
        success=True,
        message="Account created and logged in.",
        user_id=response.user.id,
        email=response.user.email,
        access_token=response.session.access_token,
        refresh_token=response.session.refresh_token,
    )


def sign_in(client: Client, email: str, password: str) -> AuthResult:
    try:
        response = client.auth.sign_in_with_password({"email": email, "password": password})
    except AuthApiError as e:
        return AuthResult(success=False, message=e.message)

    if response.user is None or response.session is None:
        return AuthResult(success=False, message="Login failed for an unknown reason.")

    return AuthResult(
        success=True,
        message="Logged in.",
        user_id=response.user.id,
        email=response.user.email,
        access_token=response.session.access_token,
        refresh_token=response.session.refresh_token,
    )


def sign_out(client: Client, access_token: str, refresh_token: str) -> None:
    """Best-effort server-side sign-out. Hydrates the client with the
    caller's own session first -- supabase-py's client doesn't otherwise
    know which session to invalidate. Clearing the local session state
    (done by the caller, in st.session_state) matters more than this
    round-trip succeeding, so failures here are swallowed."""
    try:
        client.auth.set_session(access_token, refresh_token)
        client.auth.sign_out()
    except AuthApiError:
        pass
