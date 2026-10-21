-- Projects, categories and the library (global concept search). Additive.

-- Project look (a project is a collection with kind = 'project'; `name` came with 014).
alter table collections
    add column description text,
    add column color text,
    add column icon text;

-- Categories of documents. `origin` says who made it: 'user' ones are never touched by the
-- automatic categorization, 'auto' ones are recreated / replaced by every run.
create table categories (
    id uuid primary key default gen_random_uuid(),
    name text not null check (length(btrim(name)) > 0),
    color text,
    origin text not null default 'user' check (origin in ('user', 'auto')),
    created_at timestamptz not null default now()
);
-- One name = one category, whoever made it (a user rename or assignment turns an auto one into a
-- user one, so the two can never collide).
create unique index categories_name_idx on categories (lower(name));

create table document_categories (
    document_id uuid not null references documents (id) on delete cascade,
    category_id uuid not null references categories (id) on delete cascade,
    origin text not null default 'user' check (origin in ('user', 'auto')),
    primary key (document_id, category_id)
);
create index document_categories_category_idx on document_categories (category_id);

-- Mean of the document's text-space chunk vectors (no fixed dimension: profiles coexist).
-- `doc_embedding_n` = chunks it was computed from: a re-index changes the count and the
-- vector is recomputed lazily.
alter table documents
    add column doc_embedding vector,
    add column doc_embedding_n int;
