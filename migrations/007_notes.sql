-- Answers saved as notes per chat. Content and citations are a snapshot: they survive reindex.

create table notes (
    id uuid primary key default gen_random_uuid(),
    chat_id uuid not null references chats (id) on delete cascade,
    message_id uuid,
    title text not null default '',
    content text not null,
    citations jsonb not null default '[]',
    created_at timestamptz not null default now()
);

create index notes_chat_idx on notes (chat_id, created_at desc);
