-- Granularity pass (video / audio): new segment origins -- `clip` (several consecutive frames
-- embedded as one input) and `context` (an audio parent window that only holds its children's
-- transcripts). The origin list is validated by the domain, so the check is dropped instead of
-- widened (the image pipeline adds origins of its own).
alter table chunks drop constraint if exists chunks_origin_check;
