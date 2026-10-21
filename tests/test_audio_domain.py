"""Windowing, silence snapping, empty windows and waveform peaks: pure functions."""

import pytest

from chat_pdf.domain.audio import (
    FRAME_S,
    Window,
    find_silences,
    is_empty,
    plan_windows,
    waveform_peaks,
)


def frames(*runs: tuple[float, float]) -> list[float]:
    """Levels for (seconds, dB) runs."""
    return [db for seconds, db in runs for _ in range(round(seconds / FRAME_S))]


def test_find_silences_keeps_only_runs_long_enough():
    levels = frames((1, -20), (0.5, -60), (1, -20), (0.1, -60), (1, -20), (0.4, -60))

    silences = find_silences(levels)

    assert [(round(a, 2), round(b, 2)) for a, b in silences] == [(1.0, 1.5), (3.6, 4.0)]


def test_find_silences_of_a_loud_file_is_empty():
    assert find_silences(frames((5, -10))) == []


def test_short_audio_is_a_single_window_covering_it():
    assert plan_windows(12.5, []) == [Window(0.0, 12.5)]


def test_no_windows_for_an_empty_file():
    assert plan_windows(0, []) == []


def test_windows_are_20_seconds_every_16_without_silences():
    windows = plan_windows(90, [])

    assert windows[0] == Window(0.0, 20.0)
    assert windows[1] == Window(16.0, 36.0)
    assert windows[-1].t_end == 90


def test_windows_cover_the_whole_file_and_overlap_their_neighbour():
    for duration in (21, 37.3, 60, 95, 600, 1234.5):
        windows = plan_windows(duration, [])
        assert windows[0].t_start == 0 and windows[-1].t_end == duration
        for a, b in zip(windows, windows[1:], strict=False):
            assert b.t_start < a.t_end  # overlap: a phrase on the edge appears in both
            assert b.t_start > a.t_start and b.t_end > a.t_end
        assert all(w.length >= 8 for w in windows)
        assert all(w.length <= 24.5 for w in windows)  # the tail rule can stretch the last one


def test_a_ten_minute_file_gives_about_37_windows():
    assert 36 <= len(plan_windows(600, [])) <= 38


def test_a_short_tail_is_absorbed_by_the_last_window():
    windows = plan_windows(83, [])  # nominal edge at 84 would leave a sliver otherwise

    assert windows[-1].t_end == 83
    assert windows[-1].t_start == 64.0


def test_edges_move_to_the_nearest_silence_within_three_seconds():
    silences = [(18.0, 18.6), (34.0, 35.0)]  # centers 18.3 and 34.5

    windows = plan_windows(90, silences)

    assert windows[0] == Window(0.0, 18.3)  # nominal end 20 -> silence 1.7 s away
    assert windows[2].t_start == 34.5  # nominal start 32 -> silence 2.5 s away


def test_a_silence_farther_than_three_seconds_is_ignored():
    windows = plan_windows(90, [(25.0, 26.0)])

    assert windows[0] == Window(0.0, 20.0)


def test_the_nearest_of_two_silences_wins():
    windows = plan_windows(90, [(17.0, 17.4), (21.0, 21.4)])

    assert windows[0].t_end == pytest.approx(21.2)  # 1.2 s away beats 2.8 s


def test_snapping_never_breaks_ordering_even_with_silences_everywhere():
    silences = [(i + 0.0, i + 0.5) for i in range(0, 200)]

    windows = plan_windows(200, silences)

    assert windows[-1].t_end == 200
    for a, b in zip(windows, windows[1:], strict=False):
        assert b.t_start > a.t_start and b.t_end > a.t_end
    assert all(w.length >= 8 for w in windows)


def test_pure_silence_windows_are_empty_and_quiet_speech_is_not():
    levels = frames((10, -80), (10, -30))

    assert is_empty(Window(0, 10), levels)
    assert not is_empty(Window(5, 15), levels)
    assert not is_empty(Window(10, 20), levels)


def test_waveform_has_800_bars_scaled_to_the_loudest():
    peaks = [0.1] * 1000 + [0.5] * 1000 + [0.25] * 1000

    bars = waveform_peaks(peaks)

    assert len(bars) == 800
    assert max(bars) == 1.0 and bars[0] == 0.2 and bars[-1] == 0.5
    assert all(0 <= b <= 1 for b in bars)


def test_waveform_of_a_short_file_has_fewer_bars_and_silence_stays_flat():
    assert len(waveform_peaks([0.2, 0.4, 0.8])) == 3
    assert waveform_peaks([0.0] * 10) == [0.0] * 10
    assert waveform_peaks([]) == []
