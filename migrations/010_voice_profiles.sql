-- Voice mode: speech-to-text and text-to-speech provider profiles, configured from the UI.

alter table provider_profiles drop constraint provider_profiles_kind_check;
alter table provider_profiles add constraint provider_profiles_kind_check
    check (kind in ('chat', 'embedding', 'stt', 'tts'));

alter table provider_profiles
    add column voice text,                                   -- tts: voice name, e.g. F2
    add column language text,                                -- stt/tts: ISO code, e.g. es
    add column protocol text check (protocol in ('openai', 'ollama'));  -- stt only

alter table app_settings
    add column active_stt_profile_id uuid references provider_profiles (id) on delete set null,
    add column active_tts_profile_id uuid references provider_profiles (id) on delete set null;
