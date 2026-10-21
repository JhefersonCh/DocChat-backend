-- Claim check per answer, a persona per chat and the persona an answer was written in.

alter table messages add column verification jsonb;  -- {sentences, claims}; null until requested
alter table messages add column persona text;
alter table chats add column persona text;
