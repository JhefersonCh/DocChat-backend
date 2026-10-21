-- Socratic tutor: open questions with a verified rubric, graded attempts, and a small
-- SM-2 style schedule per question (failed ones are due again right away).

create table tutor_questions (
    id uuid primary key default gen_random_uuid(),
    chat_id uuid not null references chats (id) on delete cascade,
    question text not null,
    -- [{point, quote, page, chunk_id, section}]: quotes verified against the chunks
    rubric jsonb not null,
    section text,
    page int,
    due_at timestamptz not null default now(),
    ease real not null default 2.5,
    interval_days real not null default 0,
    reps int not null default 0,
    lapses int not null default 0,
    last_score real,
    created_at timestamptz not null default now()
);

create index tutor_questions_chat_idx on tutor_questions (chat_id, due_at);

create table tutor_attempts (
    id uuid primary key default gen_random_uuid(),
    question_id uuid not null references tutor_questions (id) on delete cascade,
    answer text not null,
    points jsonb not null,
    score real not null,
    feedback text not null default '',
    follow_up text not null default '',
    created_at timestamptz not null default now()
);

create index tutor_attempts_question_idx on tutor_attempts (question_id, created_at desc);
