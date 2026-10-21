"""ffprobe / ffmpeg adapter with the subprocess mocked: parsing and error mapping."""

import asyncio
import io
import wave
from pathlib import Path

import pytest

from chat_pdf.adapters.audio.probe import FfmpegAudioProbe, parse_levels
from chat_pdf.domain.errors import InvalidInput

PATH = Path("/x/original.mp3")


def scripted(code: int, out: bytes):
    calls: list[list[str]] = []

    async def run(args, timeout):
        calls.append(args)
        return code, out

    return run, calls


def test_duration_is_read_from_ffprobe():
    run, calls = scripted(0, b"83.456000\n")

    assert asyncio.run(FfmpegAudioProbe(run).duration(PATH)) == pytest.approx(83.456)
    assert calls[0][0] == "ffprobe" and str(PATH) in calls[0]


@pytest.mark.parametrize("code,out", [(1, b""), (0, b""), (0, b"N/A\n"), (0, b"0.0\n")])
def test_duration_of_something_that_is_not_audio_is_invalid(code, out):
    run, _ = scripted(code, out)

    with pytest.raises(InvalidInput) as err:
        asyncio.run(FfmpegAudioProbe(run).duration(PATH))

    assert err.value.code == "audio_undecodable"


ASTATS = """frame:0    pts:0       pts_time:0
lavfi.astats.Overall.Peak_level=-6.020600
lavfi.astats.Overall.RMS_level=-20.0
frame:1    pts:800     pts_time:0.05
lavfi.astats.Overall.Peak_level=-inf
lavfi.astats.Overall.RMS_level=-inf
"""


def test_levels_are_parsed_into_rms_db_and_linear_peaks():
    levels = parse_levels(ASTATS)

    assert levels.frame_s == 0.05
    assert levels.rms_db[0] == -20.0 and levels.rms_db[1] == float("-inf")
    assert levels.peaks == [pytest.approx(0.5, abs=1e-3), 0.0]


def test_levels_calls_ffmpeg_and_fails_when_nothing_decodes():
    run, calls = scripted(0, ASTATS.encode())
    assert len(asyncio.run(FfmpegAudioProbe(run).levels(PATH)).rms_db) == 2
    assert "astats" in " ".join(calls[0])

    run, _ = scripted(1, b"")
    with pytest.raises(InvalidInput):
        asyncio.run(FfmpegAudioProbe(run).levels(PATH))


def test_a_window_is_cut_with_seek_and_wrapped_as_16k_mono_wav():
    pcm = b"\x01\x00" * 16000  # 1 s
    run, calls = scripted(0, pcm)

    wav = asyncio.run(FfmpegAudioProbe(run).window_wav(PATH, 16.0, 36.5))

    args = calls[0]
    assert args[args.index("-ss") + 1] == "16.000" and args[args.index("-t") + 1] == "20.500"
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.getnframes() == 16000


def test_a_failed_slice_is_invalid_input():
    run, _ = scripted(1, b"")

    with pytest.raises(InvalidInput):
        asyncio.run(FfmpegAudioProbe(run).window_wav(PATH, 0, 5))
