-- Vision provider profiles (image captions, plan §1.3 / D4) and the second ingest stage's error.

alter table provider_profiles drop constraint provider_profiles_kind_check;
alter table provider_profiles add constraint provider_profiles_kind_check
    check (kind in ('chat', 'embedding', 'stt', 'tts', 'vision'));

alter table app_settings
    add column active_vision_profile_id uuid references provider_profiles (id) on delete set null;

-- Why the descriptions stage ended partial / failed (the source itself stays `ready`).
alter table documents add column enrich_error text;
