-- Run this once in the Supabase SQL Editor (Project -> SQL Editor -> New query).
--
-- Users themselves are NOT a table here -- Supabase Auth already manages
-- that as auth.users, which everything below references.
--
-- conversations/messages tables are created now so the schema is ready,
-- but the application doesn't wire them up yet -- that's the next step.
-- documents IS fully wired this step: uploads write here, and every
-- indexed chunk in ChromaDB is now tagged with user_id and filtered by
-- it at query time. This file is the other half of that -- without RLS,
-- the app-level user_id filtering would be the ONLY thing preventing one
-- user from reading another's rows, which your original spec explicitly
-- said not to rely on ("do not simply rely on the UI to hide documents").

create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    filename text not null,
    file_size_bytes bigint,
    page_count integer,
    status text not null default 'uploading',  -- uploading|extracting|embedding|indexing|completed|failed
    chunk_count integer not null default 0,
    image_count integer not null default 0,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists documents_user_id_idx on documents(user_id);

create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    document_id uuid references documents(id) on delete set null,  -- null = searched across all of this user's documents
    title text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists conversations_user_id_idx on conversations(user_id);

create table if not exists messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references conversations(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    sources jsonb,
    created_at timestamptz not null default now()
);

create index if not exists messages_conversation_id_idx on messages(conversation_id);

-- ------------------------------------------------------------------
-- Row Level Security -- each user can only see/modify their own rows,
-- enforced by Postgres itself regardless of what the application code
-- does or forgets to do.
-- ------------------------------------------------------------------

alter table documents enable row level security;
alter table conversations enable row level security;
alter table messages enable row level security;

create policy "Users manage their own documents"
    on documents for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "Users manage their own conversations"
    on conversations for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

-- messages has no user_id column of its own -- ownership is enforced via
-- the parent conversation's user_id
create policy "Users manage messages in their own conversations"
    on messages for all
    using (exists (
        select 1 from conversations c
        where c.id = messages.conversation_id and c.user_id = auth.uid()
    ))
    with check (exists (
        select 1 from conversations c
        where c.id = messages.conversation_id and c.user_id = auth.uid()
    ));
