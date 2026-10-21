"""Composition: builds adapters and use cases. The only place that knows concrete classes."""

from dataclasses import dataclass
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from chat_pdf.adapters.audio.ffmpeg import FfmpegConverter
from chat_pdf.adapters.audio.probe import FfmpegAudioProbe
from chat_pdf.adapters.extractors import LOADERS, MEDIA_EXTRACTORS
from chat_pdf.adapters.media.encode import crop_region
from chat_pdf.adapters.persistence.artifacts import PgArtifactRepository
from chat_pdf.adapters.persistence.chat_search import PgChatSearchRepository
from chat_pdf.adapters.persistence.chats import PgChatRepository
from chat_pdf.adapters.persistence.chunks import PgChunkStore
from chat_pdf.adapters.persistence.connection import PgDatabase
from chat_pdf.adapters.persistence.documents import PgDocumentRepository
from chat_pdf.adapters.persistence.library import PgLibraryRepository
from chat_pdf.adapters.persistence.notes import PgNoteRepository
from chat_pdf.adapters.persistence.providers import PgProviderRepository
from chat_pdf.adapters.persistence.tutor import PgTutorRepository
from chat_pdf.adapters.persistence.uow import PgUnitOfWork
from chat_pdf.adapters.providers.reranking import IdentityReranker
from chat_pdf.adapters.storage.local import LocalFileStorage
from chat_pdf.adapters.video.ffmpeg import FfmpegVideoProbe
from chat_pdf.application.artifacts import (
    DeleteArtifact,
    FailInterruptedArtifacts,
    GenerateArtifact,
    ListArtifacts,
    RequestArtifact,
)
from chat_pdf.application.ask import AskQuestion
from chat_pdf.application.categories import CategoryService
from chat_pdf.application.chats import (
    CreateChat,
    DeleteChat,
    GetChat,
    ListChats,
    ListMessages,
    RemoveSource,
    UpdateChatSettings,
)
from chat_pdf.application.compatibility import CompatibilityService
from chat_pdf.application.concept_map import ExpandConceptMap
from chat_pdf.application.documents import (
    CancelIngest,
    FailInterruptedIngests,
    GetDocument,
    IngestDocument,
    OpenDocumentFile,
    ReindexDocument,
    UploadDocument,
)
from chat_pdf.application.enrich import EnrichDocument, RequestEnrichment
from chat_pdf.application.generation import GenerateOverview, RequestSummary, SuggestFollowUps
from chat_pdf.application.global_search import SearchAll
from chat_pdf.application.ingest_audio import AudioIngest
from chat_pdf.application.ingest_media import MediaIngest
from chat_pdf.application.ingest_video import VideoIngest
from chat_pdf.application.inspector import ListChunks
from chat_pdf.application.library import AskDocuments, GlobalSearch, LibraryService
from chat_pdf.application.media import GetStoryboard, GetWaveform, OpenSegmentAsset
from chat_pdf.application.notes import DeleteNote, ListNotes, RenameNote, SaveNote
from chat_pdf.application.ports import ProviderFactory
from chat_pdf.application.projects import ProjectService
from chat_pdf.application.providers import (
    CheckHealth,
    ManageProviders,
    ProbeProvider,
    SeedDefaults,
    SetActiveProfiles,
)
from chat_pdf.application.search import SearchDocument
from chat_pdf.application.slides import AssessSlides
from chat_pdf.application.transcribe import TranscribeWindows
from chat_pdf.application.tutor import (
    AnswerTutorQuestion,
    GenerateTutorQuestions,
    ListTutorQuestions,
)
from chat_pdf.application.verification import VerifyAnswer
from chat_pdf.application.voice import MAX_AUDIO_SECONDS, SpeakText, TranscribeAudio
from chat_pdf.config import Settings
from chat_pdf.crypto import Crypto
from chat_pdf.domain.config import RagConfig


@dataclass(frozen=True)
class LibraryUseCases:
    """Projects, categories and the library (global concept search)."""

    projects: ProjectService
    categories: CategoryService
    library: LibraryService
    search: GlobalSearch
    ask: AskDocuments
    compatibility: CompatibilityService


@dataclass(frozen=True)
class UseCases:
    create_chat: CreateChat
    delete_chat: DeleteChat
    remove_source: RemoveSource
    list_chats: ListChats
    get_chat: GetChat
    list_messages: ListMessages
    update_chat_settings: UpdateChatSettings
    search_document: SearchDocument
    upload_document: UploadDocument
    ingest_document: IngestDocument
    get_document: GetDocument
    open_document_file: OpenDocumentFile
    open_segment_asset: OpenSegmentAsset
    get_waveform: GetWaveform
    get_storyboard: GetStoryboard
    reindex_document: ReindexDocument
    cancel_ingest: CancelIngest
    ask_question: AskQuestion
    suggest_follow_ups: SuggestFollowUps
    verify_answer: VerifyAnswer
    request_summary: RequestSummary
    generate_overview: GenerateOverview
    request_enrichment: RequestEnrichment
    enrich_document: EnrichDocument
    manage_providers: ManageProviders
    probe_provider: ProbeProvider
    set_active_profiles: SetActiveProfiles
    seed_defaults: SeedDefaults
    fail_interrupted_ingests: FailInterruptedIngests
    check_health: CheckHealth
    request_artifact: RequestArtifact
    generate_artifact: GenerateArtifact
    list_artifacts: ListArtifacts
    delete_artifact: DeleteArtifact
    assess_slides: AssessSlides
    expand_concept_map: ExpandConceptMap
    fail_interrupted_artifacts: FailInterruptedArtifacts
    save_note: SaveNote
    list_notes: ListNotes
    rename_note: RenameNote
    delete_note: DeleteNote
    list_chunks: ListChunks
    generate_tutor_questions: GenerateTutorQuestions
    list_tutor_questions: ListTutorQuestions
    answer_tutor_question: AnswerTutorQuestion
    transcribe_audio: TranscribeAudio
    speak_text: SpeakText
    library: LibraryUseCases
    search_all: SearchAll


def build_use_cases(
    settings: Settings, rag: RagConfig, pool: AsyncConnectionPool, factory: ProviderFactory
) -> UseCases:
    providers = PgProviderRepository(pool, Crypto(settings.app_secret_key))
    chats = PgChatRepository(pool)
    docs = PgDocumentRepository(pool)
    chunks = PgChunkStore(pool)
    storage = LocalFileStorage(Path(settings.upload_dir))

    def uow() -> PgUnitOfWork:
        return PgUnitOfWork(pool)

    artifacts = PgArtifactRepository(pool)
    notes = PgNoteRepository(pool)
    tutor = PgTutorRepository(pool)
    overview = GenerateOverview(uow, providers, factory)
    probe = FfmpegAudioProbe()
    transcribe = TranscribeWindows(uow, providers, factory, storage, probe)
    enrich = EnrichDocument(uow, providers, factory, storage, audio=transcribe, crop=crop_region)
    audio = AudioIngest(probe, factory, storage)
    media = MediaIngest(MEDIA_EXTRACTORS, factory, storage)
    video = VideoIngest(FfmpegVideoProbe(), audio, factory, storage)

    lib = PgLibraryRepository(pool)
    projects = ProjectService(lib, providers, storage)
    library = LibraryUseCases(
        projects=projects,
        categories=CategoryService(lib, providers, factory),
        library=LibraryService(lib),
        search=GlobalSearch(lib, providers, chunks, IdentityReranker(), factory, rag),
        ask=AskDocuments(lib, chats, projects, storage),
        compatibility=CompatibilityService(lib, chats, docs, providers, factory),
    )

    return UseCases(
        create_chat=CreateChat(uow),
        delete_chat=DeleteChat(uow, storage),
        remove_source=RemoveSource(uow, storage),
        list_chats=ListChats(chats),
        get_chat=GetChat(chats, docs),
        list_messages=ListMessages(chats),
        update_chat_settings=UpdateChatSettings(chats, docs),
        search_document=SearchDocument(docs, providers, chunks, IdentityReranker(), factory, rag),
        upload_document=UploadDocument(uow, providers, storage),
        ingest_document=IngestDocument(
            uow, providers, factory, LOADERS, storage, rag, overview, media, enrich, audio, video
        ),
        get_document=GetDocument(docs),
        open_document_file=OpenDocumentFile(docs, storage),
        open_segment_asset=OpenSegmentAsset(chunks, storage),
        get_waveform=GetWaveform(docs, storage),
        get_storyboard=GetStoryboard(docs, chunks),
        reindex_document=ReindexDocument(docs, storage),
        cancel_ingest=CancelIngest(docs),
        ask_question=AskQuestion(
            docs, chats, providers, chunks, IdentityReranker(), factory, rag, chunks, storage
        ),
        suggest_follow_ups=SuggestFollowUps(chats, providers, factory),
        verify_answer=VerifyAnswer(chats, providers, factory),
        request_summary=RequestSummary(docs, providers),
        generate_overview=overview,
        request_enrichment=RequestEnrichment(docs, providers),
        enrich_document=enrich,
        manage_providers=ManageProviders(providers),
        probe_provider=ProbeProvider(providers, factory),
        set_active_profiles=SetActiveProfiles(providers),
        seed_defaults=SeedDefaults(
            providers, settings.seed_embedding_base_url, settings.seed_tts_base_url
        ),
        fail_interrupted_ingests=FailInterruptedIngests(docs),
        check_health=CheckHealth(PgDatabase(pool), providers, factory),
        request_artifact=RequestArtifact(chats, docs, providers, artifacts),
        generate_artifact=GenerateArtifact(artifacts, docs, chunks, providers, factory, chats, rag),
        list_artifacts=ListArtifacts(chats, artifacts),
        delete_artifact=DeleteArtifact(artifacts),
        assess_slides=AssessSlides(chats, docs, chunks, providers, factory),
        expand_concept_map=ExpandConceptMap(
            chats, docs, chunks, providers, factory, artifacts, rag
        ),
        fail_interrupted_artifacts=FailInterruptedArtifacts(artifacts),
        save_note=SaveNote(chats, notes),
        list_notes=ListNotes(chats, notes),
        rename_note=RenameNote(notes),
        delete_note=DeleteNote(notes),
        list_chunks=ListChunks(docs, chunks),
        generate_tutor_questions=GenerateTutorQuestions(
            chats, docs, chunks, providers, factory, tutor
        ),
        list_tutor_questions=ListTutorQuestions(chats, tutor),
        answer_tutor_question=AnswerTutorQuestion(providers, factory, tutor),
        transcribe_audio=TranscribeAudio(providers, factory, FfmpegConverter(MAX_AUDIO_SECONDS)),
        speak_text=SpeakText(providers, factory),
        library=library,
        search_all=SearchAll(PgChatSearchRepository(pool)),
    )
