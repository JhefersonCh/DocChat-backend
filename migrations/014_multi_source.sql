-- Multi-source chats (plan §5, phase 7). Additive: existing chats keep their one-document
-- collection untouched.

-- A collection is the unit that owns sources (documents). Today every chat owns a private one
-- (kind 'chat', 1:1). `kind = 'project'` is reserved for the future "project": ONE collection
-- shared by MANY chats (chats.collection_id is deliberately not unique). `name` is the project's
-- title; chats keep their own `title`.
alter table collections
    add column kind text not null default 'chat' check (kind in ('chat', 'project')),
    add column name text;

-- Per-chat source selection, stored as the EXCLUSIONS: a source added later is active by
-- default, and in a project every chat can mute different sources of the shared collection.
alter table chats add column disabled_sources uuid[] not null default '{}';

-- Sources of a collection in the order they were added (S1..Sn aliases follow this order).
create index documents_collection_created_idx on documents (collection_id, created_at, id);
