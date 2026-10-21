import asyncio
import io
import wave

import pytest
from fakes import (
    FakeProviderFactory,
    FakeProviderRepository,
    FakeSpeaker,
    FakeTranscriber,
    stt_profile,
    tts_profile,
)

from chat_pdf.application.providers import ManageProviders, ProbeProvider, SeedDefaults, probe_wav
from chat_pdf.application.voice import SpeakText, TranscribeAudio
from chat_pdf.domain.errors import Conflict, InvalidInput, ProviderError
from chat_pdf.domain.rules import normalize_profile_data
from chat_pdf.domain.speech import strip_for_speech


def wav(seconds: float, rate: int = 16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class PassthroughConverter:
    async def to_wav16k(self, audio):
        return audio


def transcribe_case(audio: bytes, transcriber=None, with_profile=True):
    profile = stt_profile()
    providers = FakeProviderRepository(profile, active={"stt": profile} if with_profile else {})
    factory = FakeProviderFactory(transcriber=transcriber or FakeTranscriber())
    use_case = TranscribeAudio(providers, factory, PassthroughConverter())
    return asyncio.run(use_case.execute(audio))


# --- TranscribeAudio ----------------------------------------------------------------------------


def test_transcribe_returns_the_transcript():
    stt = FakeTranscriber("cuántos días son")
    assert transcribe_case(wav(1.0), stt) == "cuántos días son"
    assert len(stt.received) == 1


def test_transcribe_without_active_profile_is_a_conflict():
    with pytest.raises(Conflict) as exc:
        transcribe_case(wav(1.0), with_profile=False)
    assert exc.value.code == "no_stt_profile"


def test_transcribe_rejects_audio_over_30_seconds():
    with pytest.raises(InvalidInput) as exc:
        transcribe_case(wav(31.0))
    assert exc.value.code == "audio_too_long"


def test_transcribe_skips_the_provider_for_a_click():
    stt = FakeTranscriber()
    assert transcribe_case(wav(0.1), stt) == ""
    assert stt.received == []


def test_transcribe_rejects_empty_upload():
    with pytest.raises(InvalidInput) as exc:
        transcribe_case(b"")
    assert exc.value.code == "empty_audio"


def test_transcribe_propagates_provider_errors():
    with pytest.raises(ProviderError):
        transcribe_case(wav(1.0), FakeTranscriber(error=ProviderError("boom")))


# --- SpeakText ----------------------------------------------------------------------------------


def speak_case(text: str, speaker=None, with_profile=True):
    profile = tts_profile()
    providers = FakeProviderRepository(profile, active={"tts": profile} if with_profile else {})
    use_case = SpeakText(providers, FakeProviderFactory(speaker=speaker or FakeSpeaker()))
    return asyncio.run(use_case.execute(text))


def test_speak_strips_markdown_and_citations_before_calling_the_provider():
    speaker = FakeSpeaker()
    clip = speak_case("El plazo es **90 días** [1][2].", speaker)
    assert speaker.spoken == ["El plazo es 90 días."]
    assert clip.mime == "audio/wav"


def test_speak_without_active_profile_is_a_conflict():
    with pytest.raises(Conflict) as exc:
        speak_case("hola", with_profile=False)
    assert exc.value.code == "no_tts_profile"


@pytest.mark.parametrize("text", ["[1]", "```\ncode\n```", "  "])
def test_speak_nothing_speakable_is_rejected(text):
    with pytest.raises(InvalidInput) as exc:
        speak_case(text)
    assert exc.value.code == "nothing_to_speak"


def test_speak_rejects_a_whole_essay():
    with pytest.raises(InvalidInput) as exc:
        speak_case("palabra " * 400)
    assert exc.value.code == "text_too_long"


# --- strip_for_speech (same vectors as frontend/src/domain/speech.test.ts) -----------------------

STRIP_CASES = [
    ("Plazo de **90 días** [1].", "Plazo de 90 días."),
    ("Ver [1, 2] y [3-4] también", "Ver y también"),
    ("# Título\n\n- uno\n- dos\n1. tres", "Título uno dos tres"),
    ("Mirá [la web](https://x.com/a) ahora", "Mirá la web ahora"),
    ("Código `x = 1` y ```py\nprint(1)\n``` fin", "Código x = 1 y fin"),
    ("Ver pág. 3 y págs. 4-5", "Ver página 3 y páginas 4-5"),
    ("Monto: 44.000.000 pesos [2]", "Monto: 44.000.000 pesos"),
    ("_cursiva_ y __negrita__", "cursiva y negrita"),
    ("| a | b |\n|---|---|\n| 1 | 2 |", "a b 1 2"),
    ("> cita textual", "cita textual"),
    ("snake_case_word se mantiene", "snake_case_word se mantiene"),
    ("ex\u00adtra\u200b ordi\u00adnario \U0001f600 fin", "extra ordinario fin"),
]


@pytest.mark.parametrize(("raw", "expected"), STRIP_CASES)
def test_strip_for_speech(raw, expected):
    assert strip_for_speech(raw) == expected


# --- Profile kinds ------------------------------------------------------------------------------


def test_stt_profile_defaults_and_ollama_drops_v1_suffix():
    data = normalize_profile_data("stt", {"base_url": "http://h:11434/v1/", "model": "m"})
    assert data["protocol"] == "ollama"
    assert data["language"] == "es"
    assert data["base_url"] == "http://h:11434"


def test_stt_openai_protocol_keeps_base_url():
    data = normalize_profile_data("stt", {"base_url": "http://h/v1", "protocol": "openai"})
    assert data["base_url"] == "http://h/v1"


def test_stt_rejects_unknown_protocol():
    with pytest.raises(InvalidInput) as exc:
        normalize_profile_data("stt", {"base_url": "http://h", "protocol": "grpc"})
    assert exc.value.code == "invalid_protocol"


def test_tts_requires_a_voice():
    with pytest.raises(InvalidInput) as exc:
        normalize_profile_data("tts", {"base_url": "http://h", "voice": "  "})
    assert exc.value.code == "voice_required"


def test_chat_profile_ignores_voice_fields():
    data = normalize_profile_data("chat", {"voice": "F2", "language": "es", "protocol": "ollama"})
    assert (data["voice"], data["language"], data["protocol"]) == (None, None, None)


def test_manage_create_and_update_normalize_voice_profiles():
    providers = FakeProviderRepository()
    manage = ManageProviders(providers)
    created = asyncio.run(
        manage.create(
            {
                "kind": "stt",
                "name": "o",
                "base_url": "http://h:11434/v1",
                "model": "m",
                "default_k": 5,
                "max_k": 15,
            }  # fmt: skip
        )
    )
    assert (created.protocol, created.base_url) == ("ollama", "http://h:11434")
    updated = asyncio.run(manage.update(created.id, {"protocol": "openai"}))
    assert updated.protocol == "openai"
    assert updated.base_url == "http://h:11434"  # untouched: nothing to strip back


# --- Probe + seed -------------------------------------------------------------------------------


def test_probe_stt_sends_a_valid_wav_and_reports_the_transcript():
    profile = stt_profile()
    stt = FakeTranscriber("")
    probe = ProbeProvider(FakeProviderRepository(profile), FakeProviderFactory(transcriber=stt))
    result = asyncio.run(probe.execute(profile.id))
    assert result.ok
    assert wave.open(io.BytesIO(stt.received[0])).getframerate() == 16_000


def test_probe_tts_synthesizes_hola():
    profile = tts_profile()
    speaker = FakeSpeaker()
    probe = ProbeProvider(FakeProviderRepository(profile), FakeProviderFactory(speaker=speaker))
    result = asyncio.run(probe.execute(profile.id))
    assert (result.ok, speaker.spoken) == (True, ["Hola"])
    assert result.audio_bytes == len(b"RIFFfake")


def test_probe_reports_provider_failure_without_raising():
    profile = tts_profile()
    factory = FakeProviderFactory(speaker=FakeSpeaker(error=ProviderError("down")))
    result = asyncio.run(
        ProbeProvider(FakeProviderRepository(profile), factory).execute(profile.id)
    )
    assert result.ok is False


def test_probe_wav_is_16k_mono():
    with wave.open(io.BytesIO(probe_wav())) as w:
        assert (w.getframerate(), w.getnchannels()) == (16_000, 1)


def test_seed_creates_local_tts_but_never_stt():
    providers = FakeProviderRepository()
    asyncio.run(SeedDefaults(providers, "http://emb/v1", "http://tts:7788/v1").execute())
    tts = asyncio.run(providers.list("tts"))
    assert [(p.voice, p.language, p.base_url) for p in tts] == [("F2", "es", "http://tts:7788/v1")]
    assert asyncio.run(providers.active("tts")) is not None
    assert asyncio.run(providers.list("stt")) == []
    asyncio.run(SeedDefaults(providers, "http://emb/v1", "http://tts:7788/v1").execute())
    assert len(asyncio.run(providers.list("tts"))) == 1  # idempotent
