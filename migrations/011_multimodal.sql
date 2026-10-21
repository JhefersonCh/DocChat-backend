-- Multimodal foundations (plan multimodal-plan.md §1): sources get a `kind`, segments (chunks) get
-- a modality + locator. 100 % additive: existing documents/chunks keep working as pdf/text.

alter table documents
    add column kind text not null default 'pdf'
        check (kind in ('pdf', 'text', 'image', 'audio', 'video')),
    add column duration_s real,        -- audio / video
    add column width int,              -- image / video
    add column height int,
    add column stage text              -- what progress_done/total measures while processing
        check (stage in ('extracting', 'embedding', 'describing', 'transcribing')),
    add column enrich_status text not null default 'skipped'
        check (enrich_status in ('pending', 'running', 'ready', 'partial', 'failed', 'skipped'));

-- Backfill: until now only pdf / txt / md could be uploaded.
update documents set kind = 'text' where mime <> 'application/pdf';

alter table chunks
    add column modality text not null default 'text'
        check (modality in ('text', 'image', 'audio', 'frame')),
    add column origin text             -- embedded_image | figure | page | tile | whole_image | window | frame
        check (origin in ('embedded_image', 'figure', 'page', 'tile', 'whole_image', 'window', 'frame')),
    add column bbox real[]             -- [x0, y0, x1, y1] normalized 0-1 over the page / image
        check (bbox is null or array_length(bbox, 1) = 4),
    add column t_start real,           -- seconds (audio / video)
    add column t_end real,
    add column asset_key text,         -- derived file name (crop / frame) inside the source's folder
    add column media_sha text,         -- dedupe / cache of repeated images
    add column text_embedding vector,  -- embedding of the description / transcript (text space)
    add column text_status text
        check (text_status in ('pending', 'ready', 'failed', 'skipped'));

-- Offsets into the document text make no sense for media segments.
alter table chunks
    alter column char_start drop not null,
    alter column char_end drop not null;

create index chunks_document_modality_idx on chunks (document_id, modality);

alter table provider_profiles
    add column supports_vision boolean not null default false,  -- chat: accepts image parts
    add column modalities text[] not null default '{text}';     -- embedding: filled by the probe
