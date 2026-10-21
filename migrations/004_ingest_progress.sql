-- Ingest progress (chunks embedded / total) and cooperative cancel flag; both only matter while
-- the document is `processing` and are cleared by every status change.

alter table documents
    add column progress_done int,
    add column progress_total int,
    add column cancel_requested boolean not null default false;
