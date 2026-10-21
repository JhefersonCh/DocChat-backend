-- Generated study artifacts (faq, quiz, mind map, ...) per chat.

create table artifacts (
    id uuid primary key default gen_random_uuid(),
    chat_id uuid not null references chats (id) on delete cascade,
    kind text not null,
    status text not null default 'pending' check (status in ('pending', 'ready', 'failed')),
    options jsonb not null default '{}',
    data jsonb,
    error text,
    created_at timestamptz not null default now()
);

create index artifacts_chat_idx on artifacts (chat_id, created_at desc);
