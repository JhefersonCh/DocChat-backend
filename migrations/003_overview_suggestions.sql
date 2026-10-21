-- Follow-up suggestions per answer, and a generated overview per document.

alter table messages add column suggestions jsonb;  -- ["question", ...]; null until requested

alter table documents
    add column summary text,
    add column suggested_questions jsonb not null default '[]',
    -- existing documents are not summarized automatically (it costs an LLM call): they stay skipped
    add column summary_status text not null default 'skipped'
        check (summary_status in ('pending', 'ready', 'failed', 'skipped'));
