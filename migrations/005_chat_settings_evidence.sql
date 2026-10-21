-- Per-chat answer style/length and a weak-evidence flag per answer.

alter table chats
    add column style text not null default 'normal'
        check (style in ('normal', 'simple', 'conciso', 'custom')),
    add column custom_instructions text,
    add column length text not null default 'normal'
        check (length in ('corta', 'normal', 'larga'));

alter table messages add column weak_evidence boolean not null default false;
