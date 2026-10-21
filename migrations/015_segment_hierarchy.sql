-- Segment hierarchy (granularity). A child segment (an object region of an image, a grid tile,
-- a fine slice of audio / video) points to its parent segment; deleting the parent drops them.
alter table chunks
    add column parent_id uuid references chunks(id) on delete cascade,
    add column region_label text;

create index chunks_parent_idx on chunks (parent_id) where parent_id is not null;
