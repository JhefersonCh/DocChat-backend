from typing import Annotated

from fastapi import APIRouter, Depends, File, Response, UploadFile
from pydantic import BaseModel, Field

from chat_pdf.adapters.http.deps import get_speak_text, get_transcribe_audio
from chat_pdf.application.voice import MAX_AUDIO_BYTES, SpeakText, TranscribeAudio
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.speech import MAX_SPEAK_CHARS

router = APIRouter(prefix="/api/v1/voice")


class TranscriptOut(BaseModel):
    text: str


class SpeakBody(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_SPEAK_CHARS * 4)


@router.post("/transcribe", response_model=TranscriptOut)
async def transcribe(
    audio: Annotated[UploadFile, File()],
    use_case: Annotated[TranscribeAudio, Depends(get_transcribe_audio)],
):
    data = await audio.read(MAX_AUDIO_BYTES + 1)  # bounded read: never buffer an unbounded body
    if len(data) > MAX_AUDIO_BYTES:
        raise InvalidInput("audio_too_large", "El audio es demasiado grande")
    return TranscriptOut(text=await use_case.execute(data))


@router.post("/speak")
async def speak(body: SpeakBody, use_case: Annotated[SpeakText, Depends(get_speak_text)]):
    clip = await use_case.execute(body.text)
    return Response(content=clip.data, media_type=clip.mime, headers={"Cache-Control": "no-store"})
