"""Pure video rules: keyframe timestamps, duplicate frames, playable codecs, storyboard."""

from uuid import uuid4

import pytest

from chat_pdf.domain.models import Chunk
from chat_pdf.domain.video import (
    VideoInfo,
    dedupe_consecutive,
    frame_ranges,
    is_duplicate_frame,
    is_playable,
    pixel_change,
    select_keyframes,
    storyboard,
)


def test_a_static_video_gets_one_frame_every_ten_seconds():
    assert select_keyframes([], 35.0) == [0.0, 8.75, 17.5, 26.25]


def test_scene_cuts_become_frames_and_long_scenes_get_filler():
    times = select_keyframes([10.0, 12.5, 40.0], 60.0)

    assert [0.0, 10.0, 12.5, 40.0] == [t for t in times if t in (0.0, 10.0, 12.5, 40.0)]
    gaps = [b - a for a, b in zip(times, [*times[1:], 60.0], strict=True)]
    assert max(gaps) <= 10.0


def test_cuts_closer_than_the_minimum_gap_to_the_last_frame_are_dropped():
    times = select_keyframes([5.0, 5.8, 6.5, 8.0], 12.0)

    assert 5.8 not in times and 6.5 not in times
    assert 5.0 in times and 8.0 in times


def test_first_frame_is_always_there_and_cuts_outside_the_video_are_ignored():
    assert select_keyframes([0.0, 99.0, -1.0], 5.0) == [0.0]
    assert select_keyframes([1.0], 0) == []


def test_the_cap_thins_evenly_and_keeps_the_first_frame():
    cuts = [float(t) for t in range(3, 3000, 3)]

    times = select_keyframes(cuts, 3000.0, cap=100)

    assert len(times) == 100 and times[0] == 0.0
    assert times == sorted(times)


def test_dedupe_drops_frames_like_the_last_kept_one():
    a, b = [1.0, 0.0], [0.0, 1.0]
    almost_a = [1.0, 0.05]

    assert dedupe_consecutive([a, almost_a, b, [0.02, 1.0], a]) == [0, 2, 4]


def test_dedupe_compares_against_the_kept_frame_not_the_previous_one():
    # each step is similar to the previous one but the drift adds up
    vectors = [[1.0, 0.1 * i] for i in range(0, 8)]

    kept = dedupe_consecutive(vectors, threshold=0.99)

    assert kept[0] == 0 and len(kept) > 1


def test_slides_with_new_text_are_not_duplicates_even_if_embeddings_agree():
    same, changed = bytes(100), bytes([200] * 10 + [0] * 90)  # 10 % of the pixels changed

    assert pixel_change(same, same) == 0.0 and pixel_change(same, changed) == 0.1
    assert pixel_change(same, b"") == 1.0 and pixel_change(same, bytes(50)) == 1.0
    assert not is_duplicate_frame(0.995, pixel_change(same, changed), gap_to_next=5.0)
    assert is_duplicate_frame(0.995, 0.0, gap_to_next=5.0)
    assert not is_duplicate_frame(0.98, 0.0, gap_to_next=5.0)  # stricter than the old 0.97


def test_a_duplicate_is_kept_when_dropping_it_would_open_a_gap_over_max_gap():
    assert is_duplicate_frame(1.0, 0.0, gap_to_next=10.0)
    assert not is_duplicate_frame(1.0, 0.0, gap_to_next=10.5)


def test_frame_ranges_last_until_the_next_kept_frame():
    assert frame_ranges([0.0, 10.0, 25.0], last_extracted=30.0, duration=60.0) == [
        (0.0, 10.0),
        (10.0, 25.0),
        (25.0, 40.0),
    ]


def test_the_last_frame_never_ends_after_the_video():
    assert frame_ranges([0.0, 20.0], last_extracted=25.0, duration=30.0)[-1] == (20.0, 30.0)


def _info(video="h264", audio="aac", container="mov,mp4,m4a,3gp,3g2,mj2", pix="yuv420p"):
    return VideoInfo(60.0, 1280, 720, video, audio, container, pix)


@pytest.mark.parametrize(
    "info,ext,playable",
    [
        (_info(), "mp4", True),
        (_info(audio=None), ".MP4", True),
        (_info(audio="mp3"), "mp4", True),
        (_info(video="hevc"), "mp4", False),
        (_info(video="mpeg4"), "mp4", False),
        (_info(pix="yuv444p"), "mp4", False),
        (_info(audio="ac3"), "mp4", False),
        (_info(), "mov", False),
        (_info(container="matroska,webm", video="h264"), "mkv", False),
        (_info(container="matroska,webm", video="vp9", audio="opus"), "webm", True),
        (_info(container="matroska,webm", video="vp8", audio="vorbis"), "webm", True),
        (_info(container="matroska,webm", video="av1", audio=None), "webm", True),
        (_info(container="matroska,webm", video="h264", audio="aac"), "webm", False),
        (_info(container="matroska,webm", video="vp9", audio="opus"), "mkv", False),
    ],
)
def test_playable_codec_decision(info, ext, playable):
    assert is_playable(info, ext) is playable


def _frame(t, ordinal, thumb=True, modality="frame"):
    return Chunk(
        ordinal=ordinal, content="", page_start=None, page_end=None, char_start=None,
        char_end=None, id=uuid4(), modality=modality, t_start=t, t_end=t + 5,
        asset_key=f"{ordinal}.webp" if thumb else None,
    )  # fmt: skip


def test_storyboard_lists_frames_with_thumbs_in_time_order():
    late, early, bare = _frame(30, 0), _frame(5, 1), _frame(10, 2, thumb=False)
    audio = _frame(0, 3, modality="audio")

    entries = storyboard([late, early, bare, audio])

    assert [(e.segment_id, e.t, e.t_end) for e in entries] == [
        (early.id, 5, 10),
        (late.id, 30, 35),
    ]
