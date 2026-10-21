-- Global keyword search over chats (messages, titles), projects and sources.
-- Substring matching that ignores case and accents: "presu" finds "Presupuesto".

create extension if not exists pg_trgm;

-- unaccent() is only STABLE, so it cannot back an index; this wrapper pins the dictionary.
create or replace function search_fold(t text) returns text
    language sql immutable parallel safe strict
    as $$ select lower(public.unaccent('public.unaccent'::regdictionary, t)) $$;

create index messages_content_trgm_idx on messages using gin (search_fold(content) gin_trgm_ops);
create index chats_title_trgm_idx on chats using gin (search_fold(title) gin_trgm_ops);
create index documents_filename_trgm_idx on documents using gin (search_fold(filename) gin_trgm_ops);
create index collections_name_trgm_idx on collections using gin (search_fold(coalesce(name, '')) gin_trgm_ops);
