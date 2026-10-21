"""The only way routers obtain use cases. Reads the composition built in `bootstrap.py`."""

from fastapi import Request

from chat_pdf.application.ask import AskQuestion
from chat_pdf.application.chats import (
    CreateChat,
    DeleteChat,
    GetChat,
    ListChats,
    ListMessages,
    RemoveSource,
    UpdateChatSettings,
)
from chat_pdf.application.documents import (
    CancelIngest,
    GetDocument,
    IngestDocument,
    OpenDocumentFile,
    ReindexDocument,
    UploadDocument,
)
from chat_pdf.application.enrich import EnrichDocument, RequestEnrichment
from chat_pdf.application.generation import GenerateOverview, RequestSummary, SuggestFollowUps
from chat_pdf.application.media import GetStoryboard, GetWaveform, OpenSegmentAsset
from chat_pdf.application.providers import (
    CheckHealth,
    ManageProviders,
    ProbeProvider,
    SetActiveProfiles,
)
from chat_pdf.application.search import SearchDocument
from chat_pdf.application.verification import VerifyAnswer
from chat_pdf.application.voice import SpeakText, TranscribeAudio


def get_create_chat(request: Request) -> CreateChat:
    return request.app.state.use_cases.create_chat


def get_delete_chat(request: Request) -> DeleteChat:
    return request.app.state.use_cases.delete_chat


def get_remove_source(request: Request) -> RemoveSource:
    return request.app.state.use_cases.remove_source


def get_list_chats(request: Request) -> ListChats:
    return request.app.state.use_cases.list_chats


def get_get_chat(request: Request) -> GetChat:
    return request.app.state.use_cases.get_chat


def get_update_chat_settings(request: Request) -> UpdateChatSettings:
    return request.app.state.use_cases.update_chat_settings


def get_search_document(request: Request) -> SearchDocument:
    return request.app.state.use_cases.search_document


def get_list_messages(request: Request) -> ListMessages:
    return request.app.state.use_cases.list_messages


def get_upload_document(request: Request) -> UploadDocument:
    return request.app.state.use_cases.upload_document


def get_ingest_document(request: Request) -> IngestDocument:
    return request.app.state.use_cases.ingest_document


def get_get_document(request: Request) -> GetDocument:
    return request.app.state.use_cases.get_document


def get_open_document_file(request: Request) -> OpenDocumentFile:
    return request.app.state.use_cases.open_document_file


def get_get_waveform(request: Request) -> GetWaveform:
    return request.app.state.use_cases.get_waveform


def get_get_storyboard(request: Request) -> GetStoryboard:
    return request.app.state.use_cases.get_storyboard


def get_open_segment_asset(request: Request) -> OpenSegmentAsset:
    return request.app.state.use_cases.open_segment_asset


def get_reindex_document(request: Request) -> ReindexDocument:
    return request.app.state.use_cases.reindex_document


def get_cancel_ingest(request: Request) -> CancelIngest:
    return request.app.state.use_cases.cancel_ingest


def get_ask_question(request: Request) -> AskQuestion:
    return request.app.state.use_cases.ask_question


def get_suggest_follow_ups(request: Request) -> SuggestFollowUps:
    return request.app.state.use_cases.suggest_follow_ups


def get_verify_answer(request: Request) -> VerifyAnswer:
    return request.app.state.use_cases.verify_answer


def get_request_summary(request: Request) -> RequestSummary:
    return request.app.state.use_cases.request_summary


def get_generate_overview(request: Request) -> GenerateOverview:
    return request.app.state.use_cases.generate_overview


def get_manage_providers(request: Request) -> ManageProviders:
    return request.app.state.use_cases.manage_providers


def get_probe_provider(request: Request) -> ProbeProvider:
    return request.app.state.use_cases.probe_provider


def get_set_active_profiles(request: Request) -> SetActiveProfiles:
    return request.app.state.use_cases.set_active_profiles


def get_check_health(request: Request) -> CheckHealth:
    return request.app.state.use_cases.check_health


def get_transcribe_audio(request: Request) -> TranscribeAudio:
    return request.app.state.use_cases.transcribe_audio


def get_speak_text(request: Request) -> SpeakText:
    return request.app.state.use_cases.speak_text


def get_request_enrichment(request: Request) -> RequestEnrichment:
    return request.app.state.use_cases.request_enrichment


def get_enrich_document(request: Request) -> EnrichDocument:
    return request.app.state.use_cases.enrich_document
