create extension if not exists vector;

create table provider_profiles (
    id uuid primary key default gen_random_uuid(),
    kind text not null check (kind in ('chat', 'embedding')),
    name text not null,
    base_url text not null,
    model text not null,
    api_key_enc bytea,
    dimensions int,
    query_prefix text,
    document_prefix text,
    supports_tools boolean not null default false,
    -- k / tool-call defaults mirror DEFAULT_K, MAX_K, MAX_TOOL_CALLS in domain/providers.py
    default_k int not null default 5,
    max_k int not null default 15,
    max_tool_calls int not null default 3,
    created_at timestamptz not null default now()
);

create table app_settings (
    id boolean primary key default true check (id),
    active_chat_profile_id uuid references provider_profiles (id) on delete set null,
    active_embedding_profile_id uuid references provider_profiles (id) on delete set null
);

create table collections (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid,
    created_at timestamptz not null default now()
);

create table chats (
    id uuid primary key default gen_random_uuid(),
    collection_id uuid not null references collections (id) on delete cascade,
    title text not null default '',
    owner_id uuid,
    created_at timestamptz not null default now()
);

create table documents (
    id uuid primary key default gen_random_uuid(),
    collection_id uuid not null references collections (id) on delete cascade,
    filename text not null,
    mime text not null,
    sha256 text not null,
    status text not null default 'pending'
        check (status in ('pending', 'processing', 'ready', 'failed')),
    error text,
    page_count int,
    embedding_profile_id uuid not null references provider_profiles (id) on delete restrict,
    created_at timestamptz not null default now()
);

create table chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents (id) on delete cascade,
    ordinal int not null,
    content text not null,
    page_start int,
    page_end int,
    char_start int not null,
    char_end int not null,
    embedding vector  -- no fixed dimension: profiles with different dims coexist
);

create index chunks_document_id_idx on chunks (document_id);

create table messages (
    id uuid primary key default gen_random_uuid(),
    chat_id uuid not null references chats (id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    citations jsonb,
    searches jsonb,  -- tool searches made for this answer: [{query, k}]
    created_at timestamptz not null default now()
);

create index messages_chat_id_created_at_idx on messages (chat_id, created_at);
