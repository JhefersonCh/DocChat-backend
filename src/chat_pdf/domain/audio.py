"""Audio segmentation (plan D9, §2.4): 20 s windows with 4 s of overlap whose edges are moved to
the nearest silence, plus the waveform. Pure: it works on loudness levels, never on samples."""

from dataclasses import dataclass

WINDOW_S = 20.0  # plan_windows defaults (generic); the ingest uses the FINE_* values below
OVERLAP_S = 4.0
SNAP_S = 3.0  # an edge moves to a silence at most this far away
MIN_WINDOW_S = 8.0  # a snapped window never gets shorter than this (except a short file)
FINE_WINDOW_S = 9.0  # retrieval windows: ~9 s, 50 % overlap
FINE_OVERLAP_S = 4.5
FINE_SNAP_S = 1.5
FINE_MIN_S = 5.0
PARENT_S = 30.0  # context windows: children grouped by midpoint into blocks of this length
CHAIN_TOLERANCE_S = 1.5  # transcripts chained into a parent may overlap this much
MAX_AUDIO_S = 3 * 3600.0  # longer sources are refused (a fine window every 4.5 s costs ~1 s)
FRAME_S = 0.05  # loudness resolution
SILENCE_DB = -40.0  # RMS below this is "silence" for snapping
MIN_SILENCE_S = 0.3
EMPTY_DB = -45.0  # a window whose every frame is below this is dropped (plan: pure silence)
WAVEFORM_POINTS = 800

Silence = tuple[float, float]  # (start s, end s)


@dataclass(frozen=True)
class Window:
    t_start: float
    t_end: float

    @property
    def length(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class AudioLevels:
    """Loudness of a decoded file, one value per `frame_s` frame."""

    frame_s: float
    rms_db: list[float]  # per-frame RMS in dB (-inf-like for digital silence)
    peaks: list[float]  # per-frame peak amplitude, linear 0-1


def find_silences(
    rms_db: list[float],
    frame_s: float = FRAME_S,
    threshold_db: float = SILENCE_DB,
    min_s: float = MIN_SILENCE_S,
) -> list[Silence]:
    """Runs of frames below `threshold_db` lasting at least `min_s`."""
    out: list[Silence] = []
    run_start: int | None = None
    for i, level in enumerate([*rms_db, 0.0]):  # the sentinel closes a trailing run
        quiet = level < threshold_db
        if quiet and run_start is None:
            run_start = i
        elif not quiet and run_start is not None:
            if (i - run_start) * frame_s >= min_s:
                out.append((run_start * frame_s, i * frame_s))
            run_start = None
    return out


def _snap(target: float, silences: list[Silence], reach: float) -> float:
    """Middle of the silence nearest to `target` (within `reach`), else `target` itself."""
    best, best_gap = target, reach + 1e-9
    for start, end in silences:
        middle = (start + end) / 2
        gap = abs(middle - target)
        if gap < best_gap:
            best, best_gap = middle, gap
    return best


def plan_windows(
    duration: float,
    silences: list[Silence],
    size: float = WINDOW_S,
    overlap: float = OVERLAP_S,
    reach: float = SNAP_S,
    min_window: float = MIN_WINDOW_S,
) -> list[Window]:
    """Windows covering [0, duration]: nominal edges every `size - overlap` seconds, each moved
    to the nearest silence. The first starts at 0 and the last ends at `duration`; a tail
    shorter than the overlap is absorbed by the last window instead of making a sliver."""
    if duration <= 0:
        return []
    step = size - overlap
    windows: list[Window] = []
    k = 0
    while True:
        nominal_start = k * step
        start = 0.0 if k == 0 else _snap(nominal_start, silences, reach)
        if windows:  # edges only move forward, and a window must overlap its predecessor
            start = min(max(start, windows[-1].t_start + min_window / 2), windows[-1].t_end)
        nominal_end = nominal_start + size
        if duration - nominal_end <= overlap:
            end = duration
        else:
            end = _snap(nominal_end, silences, reach)
            end = min(max(end, start + min_window), duration)
        windows.append(Window(round(start, 3), round(end, 3)))
        if end >= duration:
            return windows
        k += 1


def is_empty(window: Window, rms_db: list[float], frame_s: float = FRAME_S) -> bool:
    """True when every frame of the window is below the pure-silence floor."""
    first = int(window.t_start / frame_s)
    last = max(int(window.t_end / frame_s), first + 1)
    frames = rms_db[first:last]
    return all(level < EMPTY_DB for level in frames)


def waveform_peaks(frame_peaks: list[float], points: int = WAVEFORM_POINTS) -> list[float]:
    """`points` bars (or fewer for a very short file): the loudest peak of each slice, scaled so
    the highest is 1. A silent file stays flat at 0."""
    if not frame_peaks:
        return []
    count = min(points, len(frame_peaks))
    bars = [
        max(frame_peaks[i * len(frame_peaks) // count : (i + 1) * len(frame_peaks) // count])
        for i in range(count)
    ]
    top = max(bars)
    return [round(b / top, 3) for b in bars] if top > 0 else [0.0] * count


@dataclass(frozen=True)
class ParentWindow:
    t_start: float
    t_end: float
    children: tuple[int, ...]  # indexes into the fine windows, in time order


def plan_parents(windows: list[Window], parent_s: float = PARENT_S) -> list[ParentWindow]:
    """Group fine windows into ~`parent_s` context windows (by the window's midpoint). Each
    parent spans from its first child's start to its last child's end."""
    groups: dict[int, list[int]] = {}
    for i, w in enumerate(windows):
        groups.setdefault(int((w.t_start + w.t_end) / 2 // parent_s), []).append(i)
    return [
        ParentWindow(windows[idx[0]].t_start, max(windows[i].t_end for i in idx), tuple(idx))
        for _, idx in sorted(groups.items())
    ]


def context_text(
    parts: list[tuple[float, float, str]], tolerance: float = CHAIN_TOLERANCE_S
) -> str:
    """Transcript of a parent: its children's (t_start, t_end, text) chained in time order,
    skipping a child that mostly repeats the previous one (50 % overlap would say everything
    twice). Children without text never take a place in the chain."""
    chain: list[str] = []
    last_end = float("-inf")
    for start, end, text in sorted(parts):
        if text.strip() and start >= last_end - tolerance:
            chain.append(text.strip())
            last_end = end
    return " ".join(chain)
