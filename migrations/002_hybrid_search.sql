-- Section-aware chunks + full-text search for hybrid retrieval (vector + FTS fused by RRF).

create extension if not exists unaccent;

-- Spanish stemming over unaccented words: "geográfica" and "geografica" match.
create text search configuration es_unaccent (copy = spanish);
alter text search configuration es_unaccent
    alter mapping for hword, hword_part, word with unaccent, spanish_stem;

alter table chunks
    add column section text,      -- title of the heading the chunk falls under
    add column index_text text;   -- normalized text that is embedded and indexed (content stays verbatim)

-- Generated, so it can never drift from the text. Chunks from before this migration have no
-- index_text and fall back to content until they are re-indexed.
alter table chunks add column tsv tsvector generated always as (
    setweight(to_tsvector('es_unaccent', coalesce(section, '')), 'A')
    || setweight(to_tsvector('es_unaccent', coalesce(index_text, content)), 'B')
) stored;

create index chunks_tsv_idx on chunks using gin (tsv);
