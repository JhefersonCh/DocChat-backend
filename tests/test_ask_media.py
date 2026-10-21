"""Phase 3 through AskQuestion: modality-aware search, find_all with vision verification and its
verdict cache, media in the answer prompt, citations with kind/bbox."""

import asyncio
from dataclasses import dataclass, replace
from uuid import uuid4

from fakes import (
    FakeChat,
    FakeChatRepository,
    FakeChunkStore,
    FakeDb,
    FakeDocumentRepository,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeVision,
    IdentityReranker,
    chat_profile,
    embedding_profile,
    vision_profile,
)

from chat_pdf.adapters.storage.local import LocalFileStorage
from chat_pdf.application.ask import AskQuestion
from chat_pdf.application.ports import ChatResult, ToolCall
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.models import Chunk


def _image(i, caption, native, page=1, bbox=None, similarity=None, origin="embedded_image"):
    return Chunk(
        i, caption, page, page, None, None, id=uuid4(), modality="image", origin=origin,
        bbox=bbox or [0.1, 0.55, 0.45, 0.9], asset_key=f"{i}.webp", text_status="ready",
        native_similarity=native, similarity=similarity,
    )  # fmt: skip


TEXT = Chunk(0, "El informe habla de mascotas.", 1, 1, 0, 30, id=uuid4(), similarity=0.7)
DOG = _image(1, "[foto] Un can marrón en el pasto", 0.80, similarity=0.7)
CAT = _image(
    2, "[foto] Un gato sobre un sofá", 0.78, page=2, bbox=[0.2, 0.1, 0.8, 0.5], similarity=0.64
)
CHART = _image(3, "[gráfico] Barras de ventas", 0.55, page=3, bbox=[0.1, 0.1, 0.9, 0.6])
MEDIA = [TEXT, DOG, CAT, CHART]


class KeyedVision(FakeVision):
    """Answers by which picture it is shown: checks run concurrently, so call order is free."""

    def __init__(self, **answers):
        super().__init__()
        self.by_image = {f"IMG{k}.webp".encode(): v for k, v in answers.items()}

    async def describe(self, image, prompt):
        self.images.append(image)
        self.prompts.append(prompt)
        answer = self.by_image.get(image, "no")
        if isinstance(answer, Exception):
            raise answer
        return answer


def _tool(name, **args):
    return ChatResult(None, [ToolCall("c", name, {"query": "perro", **args})])


@dataclass
class Setup:
    use_case: AskQuestion
    chat_id: object
    db: FakeDb
    store: FakeChunkStore
    chat: FakeChat
    vision: FakeVision

    def ask(self, question="¿Dónde hay un perro?"):
        async def go():
            return [e async for e in await self.use_case.execute(self.chat_id, question)]

        return asyncio.run(go())


def _setup(tmp_path, chat=None, corpus=MEDIA, vision=None, chat_kw=None, with_vision_profile=True):
    chat_p = chat_profile(**{"supports_tools": True, "max_tool_calls": 3, **(chat_kw or {})})
    emb = embedding_profile()
    active = {"chat": chat_p}
    if with_vision_profile:
        active["vision"] = vision_profile()
    db = FakeDb()
    chat_row = db.add_chat()
    doc = db.add_document(chat_row.id, emb.id)
    db.docs[doc.id] = replace(doc, kind="pdf", filename="informe.pdf", page_count=3)
    db.chunks[doc.id] = list(corpus)
    storage = LocalFileStorage(tmp_path)
    for c in corpus:
        if c.asset_key:
            asyncio.run(
                storage.save_asset(doc.id, "assets", c.asset_key, b"IMG" + c.asset_key.encode())
            )
    chat = chat or FakeChat()
    vision = vision or FakeVision()
    store = FakeChunkStore(db)
    factory = FakeProviderFactory(chat, vision=vision)
    use_case = AskQuestion(
        FakeDocumentRepository(db), FakeChatRepository(db),
        FakeProviderRepository(chat_p, emb, *([active["vision"]] if with_vision_profile else []),
                               active=active),
        store, IdentityReranker(), factory, RagConfig(), store, storage,
    )  # fmt: skip
    return Setup(use_case, chat_row.id, db, store, chat, vision)


def _names(events):
    return [name for name, _ in events]


def test_find_all_lists_only_the_vision_confirmed_occurrences(tmp_path):
    # candidates by recall: dog (0.80), cat (0.78); the chart (0.55) is noise. Strongest first.
    chat = FakeChat(rounds=[_tool("find_all")])
    setup = _setup(
        tmp_path,
        chat,
        vision=KeyedVision(**{"1": "Sí, abajo a la izquierda", "2": "No, es un gato"}),
    )

    events = setup.ask()

    search = events[0][1]
    assert events[0][0] == "search" and search["exhaustive"] is True and search["hits"] == 1
    assert search["rankings"] == {"checked": 1, "discarded": 1, "asked": 2}
    [citation] = events[1][1]
    assert citation.chunk_id == DOG.id and citation.kind == "image"
    assert citation.bbox == DOG.bbox and citation.has_thumb is True and citation.page == 1
    assert len(setup.vision.prompts) == 2 and "«perro»" in setup.vision.prompts[0]
    assert sorted(setup.vision.images) == [b"IMG1.webp", b"IMG2.webp"]
    [persisted] = setup.db.exchanges[0]["searches"]
    assert persisted.exhaustive is True and persisted.found[0]["bbox"] == DOG.bbox
    tool_result = chat.tool_messages[-1][-1] if len(chat.tool_messages) > 1 else None
    assert tool_result is None or "ocurrencias" in tool_result["content"]


def test_find_all_tells_the_model_when_nothing_matches(tmp_path):
    chat = FakeChat(rounds=[_tool("find_all", query="dinosaurio"), ChatResult("listo")])
    setup = _setup(tmp_path, chat, vision=FakeVision("No", "No"))

    events = setup.ask("¿Hay un dinosaurio?")

    assert events[0][1]["hits"] == 0 and events[1][1] == []  # no citations
    result = chat.tool_messages[1][-1]
    assert result["role"] == "tool" and "ninguna ocurrencia" in result["content"]
    assert "descartaron 2" in result["content"]
    assert events[-1][1]["weak_evidence"] is True  # nothing supports an answer


def test_verdicts_are_cached_per_segment_and_normalized_query(tmp_path):
    chat = FakeChat(rounds=[_tool("find_all"), _tool("find_all", query="¿Un PERRO?")])
    setup = _setup(tmp_path, chat, vision=KeyedVision(**{"1": "sí", "2": "no"}))

    setup.ask()
    first_calls = len(setup.vision.prompts)
    chat.rounds = [_tool("find_all", query="perro!")]  # same words, different spelling noise
    events = setup.ask("¿y ahora?")

    assert first_calls == 2
    assert len(setup.vision.prompts) == 2  # the second search paid nothing
    assert events[0][1]["rankings"]["asked"] == 0 and events[0][1]["hits"] == 1
    assert set(setup.store.verdict_cache.values()) == {True, False}


def test_a_different_query_is_asked_again(tmp_path):
    chat = FakeChat(rounds=[_tool("find_all")])
    setup = _setup(tmp_path, chat, vision=KeyedVision(**{"1": "sí", "2": "no"}))
    setup.ask()
    chat.rounds = [_tool("find_all", query="gato")]

    setup.ask("¿y un gato?")

    # the dog is checked again for «gato»; the cat's own caption makes it a clear match
    assert len(setup.vision.prompts) == 3 and "«gato»" in setup.vision.prompts[-1]


def test_without_a_vision_profile_candidates_come_back_labeled(tmp_path):
    chat = FakeChat(rounds=[_tool("find_all"), ChatResult("listo")])
    setup = _setup(tmp_path, chat, with_vision_profile=False)

    events = setup.ask()

    assert setup.vision.prompts == []
    assert [c.chunk_id for c in events[1][1]] == [DOG.id, CAT.id]  # reading order: page 1, 2
    result = chat.tool_messages[1][-1]["content"]
    assert "sin verificación visual" in result and "[confianza" in result


def test_vision_failure_degrades_to_unverified_instead_of_failing(tmp_path):
    from chat_pdf.domain.errors import ProviderError

    chat = FakeChat(rounds=[_tool("find_all"), ChatResult("listo")])
    setup = _setup(tmp_path, chat, vision=KeyedVision(**{"1": ProviderError("boom"), "2": "sí"}))

    events = setup.ask()

    assert _names(events)[-1] == "done"
    assert len(events[1][1]) >= 1
    assert setup.store.verdict_cache and True in setup.store.verdict_cache.values()


def test_modality_hint_reweights_and_filters_the_search(tmp_path):
    chat = FakeChat(rounds=[_tool("search_document", k=5, modality="imagen")])
    setup = _setup(tmp_path, chat)

    events = setup.ask("¿Qué muestra el gráfico?")

    search = events[0][1]
    assert search["modality"] == "imagen"
    assert setup.store.last_modalities == ("image", "frame")
    assert search["rankings"]["image"] == 3 and search["rankings"]["vector"] == 3
    assert search["found"][0]["kind"] == "image" and search["found"][0]["has_thumb"] is True


def test_unknown_modality_means_auto_and_text_hint_skips_native_rankings(tmp_path):
    chat = FakeChat(rounds=[_tool("search_document", k=5, modality="holograma")])
    setup = _setup(tmp_path, chat)
    assert setup.ask()[0][1]["modality"] == "auto" and setup.store.last_modalities is None

    chat = FakeChat(rounds=[_tool("search_document", k=5, modality="texto")])
    setup = _setup(tmp_path, chat)
    search = setup.ask()[0][1]
    assert search["rankings"]["image"] == 0 and search["rankings"]["audio"] == 0


def test_planner_gets_catalog_find_all_and_modality_only_for_media_documents(tmp_path):
    media_chat = FakeChat()
    _setup(tmp_path, media_chat).ask()
    text_chat = FakeChat()
    _setup(tmp_path / "t", text_chat, corpus=[TEXT]).ask()

    system = media_chat.tool_systems[0]
    assert "S1 · informe.pdf · PDF 3 págs · 3 imágenes (3 descritas)" in system
    assert "find_all" in system
    names = [t["function"]["name"] for t in media_chat.tool_specs[0]]
    assert names == ["search_document", "find_all"]
    assert "modality" in media_chat.tool_specs[0][0]["function"]["parameters"]["properties"]
    assert "find_all" not in text_chat.tool_systems[0] and "S1 ·" not in text_chat.tool_systems[0]
    assert [t["function"]["name"] for t in text_chat.tool_specs[0]] == ["search_document"]


def test_answer_prompt_describes_media_sources_with_locators(tmp_path):
    chat = FakeChat()
    setup = _setup(tmp_path, chat)

    setup.ask()

    [system, user] = chat.streamed[0]
    assert isinstance(user["content"], str)  # DeepSeek-style chat: text only
    assert "(imagen · pág. 1, abajo a la izquierda) [foto] Un can marrón" in user["content"]
    assert "no inventes detalles" in system["content"]


def test_vision_capable_chat_gets_up_to_three_images_in_the_user_message_only(tmp_path):
    many = [TEXT, *[_image(i, f"foto {i}", 0.8, page=i) for i in range(1, 6)]]
    chat = FakeChat()
    setup = _setup(tmp_path, chat, corpus=many, chat_kw={"supports_vision": True, "default_k": 5})

    setup.ask()

    [system, user] = chat.streamed[0]
    assert isinstance(system["content"], str)
    images = [p for p in user["content"] if p["type"] == "image_url"]
    assert len(images) == 3 and images[0]["image_url"]["url"].startswith("data:image/webp;base64,")
    assert "adjuntan" in system["content"]


def test_weak_evidence_for_images_uses_the_image_threshold(tmp_path):
    weak = [_image(1, "borroso", 0.63)]  # over the text 0.60, under the image 0.66
    strong = [_image(1, "nítido", 0.72)]

    assert _setup(tmp_path / "a", corpus=weak).ask()[-1][1]["weak_evidence"] is True
    assert _setup(tmp_path / "b", corpus=strong).ask()[-1][1]["weak_evidence"] is False


def test_text_only_documents_keep_their_events_and_prompt(tmp_path):
    chat = FakeChat()
    events = _setup(tmp_path, chat, corpus=[TEXT]).ask("¿qué dice?")

    assert _names(events) == ["search", "citations", "token", "done"]
    assert events[0][1]["modality"] == "auto" and events[0][1]["exhaustive"] is False
    assert isinstance(chat.streamed[0][1]["content"], str)
    assert "imagen" not in chat.streamed[0][0]["content"].lower()


def test_clear_band_is_accepted_without_asking_the_vision_model(tmp_path):
    barking = _image(1, "[foto] Un perro ladrando", 0.80)  # native over threshold + keyword
    chat = FakeChat(rounds=[_tool("find_all"), ChatResult("listo")])
    setup = _setup(tmp_path, chat, corpus=[TEXT, barking])

    events = setup.ask()

    assert setup.vision.prompts == []
    assert [c.chunk_id for c in events[1][1]] == [barking.id]
    assert "[confianza alta]" in chat.tool_messages[1][-1]["content"]
