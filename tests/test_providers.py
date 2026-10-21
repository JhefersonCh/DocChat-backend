import asyncio

import pytest
from fakes import (
    FakeChat,
    FakeProviderFactory,
    FakeProviderRepository,
    chat_profile,
    embedding_profile,
)

from chat_pdf.application.ports import ChatResult, ToolCall
from chat_pdf.application.providers import ManageProviders, ProbeProvider
from chat_pdf.domain.errors import Conflict


@pytest.mark.parametrize(
    ("rounds", "expected"),
    [([ChatResult(None, [ToolCall("c", "ping", {})])], True), ([], False)],
)
def test_probe_detects_and_persists_tool_support(rounds, expected):
    profile = chat_profile()
    providers = FakeProviderRepository(profile)
    probe = ProbeProvider(providers, FakeProviderFactory(chat=FakeChat(rounds=rounds)))

    result = asyncio.run(probe.execute(profile.id))

    assert result.supports_tools is expected
    assert providers.profiles[profile.id].supports_tools is expected


def test_embedding_profile_in_use_rejects_changes_to_the_vector_space():
    profile = embedding_profile()
    providers = FakeProviderRepository(profile)
    providers.used.add(profile.id)
    manage = ManageProviders(providers)

    with pytest.raises(Conflict) as exc:
        asyncio.run(manage.update(profile.id, {"model": "other"}))
    assert exc.value.code == "provider_in_use"

    asyncio.run(manage.update(profile.id, {"name": "renamed"}))  # harmless fields still allowed
    assert providers.profiles[profile.id].name == "renamed"


def test_vision_probe_sends_an_image_and_reports_the_caption():
    from fakes import FakeVision, vision_profile

    profile = vision_profile()
    vision = FakeVision("Un degradado azul")
    probe = ProbeProvider(FakeProviderRepository(profile), FakeProviderFactory(vision=vision))

    result = asyncio.run(probe.execute(profile.id))

    assert (result.ok, result.caption) == (True, "Un degradado azul")
    assert vision.images[0].startswith(b"\x89PNG")


def test_vision_probe_reports_an_empty_answer_as_failure():
    from fakes import FakeVision, vision_profile

    from chat_pdf.domain.errors import EmptyCompletion

    profile = vision_profile()
    probe = ProbeProvider(
        FakeProviderRepository(profile),
        FakeProviderFactory(vision=FakeVision(EmptyCompletion("vacía"))),
    )

    result = asyncio.run(probe.execute(profile.id))

    assert result.ok is False and "vacía" in result.detail
