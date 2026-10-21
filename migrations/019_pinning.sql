-- Pinned chats and projects (sidebar): null = not pinned; most recently pinned sorts first.
alter table chats add column if not exists pinned_at timestamptz;
alter table collections add column if not exists pinned_at timestamptz;
