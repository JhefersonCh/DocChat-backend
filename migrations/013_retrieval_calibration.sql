-- Multimodal retrieval (plan §3.2, §3.5): per-modality calibration on the embedding profile and
-- the cache of vision verdicts of the exhaustive search (find_all).

-- Embedding profiles: {"text": {"floor", "threshold"}, "image": {...}, "audio": {...}}. NULL means
-- the code defaults (the spike's values); scripts/calibrate_modalities.py writes it.
alter table provider_profiles add column calibration jsonb;

-- "Does this segment show X?" asked once per (segment, normalized query): repeated find_all calls
-- do not pay the vision model again. Dropped with the segment (a re-index recreates the chunks).
create table segment_verdicts (
    chunk_id   uuid        not null references chunks (id) on delete cascade,
    query_key  text        not null,
    verdict    boolean     not null,
    created_at timestamptz not null default now(),
    primary key (chunk_id, query_key)
);
