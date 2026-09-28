"""The Claude-like feel of a reply being written: rotating status words, a steady reveal of
the text, new words fading in, and a background reader so the page never freezes.

The model sends text in irregular bursts, and nothing arrives at all while it loads or
reads the prompt. ``StreamPump`` reads the stream on a background thread (it only collects
text; all drawing stays on Streamlit's thread), and the page reveals the buffered text at an
even pace with ``reveal_count``, so it neither stalls nor jumps.
"""

import html
import math
import queue
import re
import threading
from collections.abc import Iterable, Sequence

# Words for what each step is doing, rotated while it runs. Our own list, in the spirit of
# a coding assistant's status line.
VERBS: dict[str, tuple[str, ...]] = {
    "embed": ("Encoding", "Vectorizing", "Distilling"),
    "search": ("Searching", "Scouring", "Sifting", "Rummaging"),
    "rerank": ("Weighing", "Ranking", "Sorting"),
    "read": ("Reading", "Digesting", "Absorbing"),
    "think": (
        "Thinking",
        "Pondering",
        "Mulling",
        "Incubating",
        "Churning",
        "Percolating",
        "Brewing",
        "Noodling",
    ),
    "write": ("Writing", "Composing", "Crafting", "Weaving"),
    "cite": ("Connecting", "Cross-checking", "Verifying"),
}
# Words to fade: runs without Markdown syntax (a span around "**", "|", "#" or "`" breaks it).
_TOKEN = re.compile(r"[^\s*_`\[\]|#>~]+")
_LINE_MARKER = re.compile(r"[-+*]|\d+[.)]")  # list markers must stay bare at a line start
_EPSILON = 1e-9


def verb(step: str, elapsed_s: float, every_s: float = 1.5) -> str:
    """The word for ``step`` after ``elapsed_s`` seconds, changing every ``every_s``."""
    words = VERBS.get(step)
    if words is None:
        raise ValueError(f"unknown step: {step}")
    return words[int(elapsed_s // every_s) % len(words)]


def status_line(
    step: str, elapsed_s: float, *, tokens: int = 0, detail: str = "", every_s: float = 1.5
) -> str:
    """E.g. "Pondering… (12s · ↓ 75 tokens)": the word, then time and tokens once known."""
    text = verb(step, elapsed_s, every_s) + (f" {detail}" if detail else "") + "…"
    meta = []
    if elapsed_s >= 1:
        meta.append(f"{int(elapsed_s)}s")
    if tokens:
        meta.append(f"↓ {tokens:,} tokens")
    return text + (f" ({' · '.join(meta)})" if meta else "")


def reveal_count(
    shown: int, available: int, dt: float, *, min_cps: float, catch_up_s: float | None
) -> int:
    """How many characters to show after ``dt`` seconds.

    At least ``min_cps`` characters per second, and fast enough to empty the backlog within
    ``catch_up_s`` (None = a fixed pace, for slow motion). Never more than has arrived.
    """
    if available <= shown:
        return shown
    step = math.ceil(min_cps * dt - _EPSILON)
    if catch_up_s is not None:
        step = max(step, math.ceil((available - shown) * dt / catch_up_s - _EPSILON))
    return min(available, shown + max(1, step))


def steps_html(lines: Sequence[str]) -> str:
    """The finished steps of a reply as a timeline: a thin line on the left, a small green dot
    per step (like a coding assistant's tool calls)."""
    if not lines:
        return ""
    rows = "".join(f'<div class="rag-step">{html.escape(line)}</div>' for line in lines)
    return f'<div class="rag-timeline">{rows}</div>'


def snap_to_word(text: str, shown: int, max_ahead: int = 12) -> int:
    """Move ``shown`` to the end of the word it cuts, so text appears a word at a time."""
    if shown <= 0 or shown >= len(text) or text[shown].isspace() or text[shown - 1].isspace():
        return shown
    gap = re.search(r"\s", text[shown : shown + max_ahead])
    return shown + gap.start() if gap else shown


def fade_levels(
    reveals: Sequence[tuple[int, float]], now: float, fade_s: float
) -> tuple[tuple[int, float], ...]:
    """(start index, progress 0..1) for every revealed piece still settling into focus."""
    levels = []
    for start, revealed_at in reveals:
        progress = (now - revealed_at) / fade_s
        if progress < 1:
            levels.append((start, max(0.0, progress)))
    return tuple(levels)


def fade_style(progress: float) -> str:
    """Faint and blurred when new, sharp when settled (the way a chat assistant's text lands)."""
    if progress >= 1:
        return ""
    return f"opacity:{0.2 + 0.8 * progress:.2f};filter:blur({3 * (1 - progress):.1f}px)"


def progress_at(index: int, levels: Sequence[tuple[int, float]]) -> float | None:
    """How settled the character at ``index`` is; None when it has fully settled."""
    found = None
    for start, progress in levels:
        if start > index:
            break
        found = progress
    return found


def fade_markdown_text(text: str, offset: int, levels: Sequence[tuple[int, float]]) -> str:
    """Escape ``text`` (``&`` and ``<``, as for Markdown) and fade its newest words.

    ``offset`` is where ``text`` starts in the whole answer. Markdown syntax, list markers at
    a line start and anything inside inline code stay untouched, so the markup still renders.
    """
    if not levels or offset + len(text) <= levels[0][0]:
        return _escape(text)
    out, position = [], 0
    for match in _TOKEN.finditer(text):
        start, end = match.span()
        out.append(_escape(text[position:start]))
        position = end
        token = match.group()
        progress = progress_at(offset + start, levels)
        before = text[text.rfind("\n", 0, start) + 1 : start]
        if (
            progress is None
            or progress >= 1
            or before.count("`") % 2 == 1
            or (not before.strip() and _LINE_MARKER.fullmatch(token))
        ):
            out.append(_escape(token))
        else:
            out.append(f'<span style="{fade_style(progress)}">{_escape(token)}</span>')
    out.append(_escape(text[position:]))
    return "".join(out)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


class StreamPump:
    """Read a text stream on a background thread; the page polls it without blocking."""

    def __init__(self, stream: Iterable[str]) -> None:
        self._pieces: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._done = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, args=(stream,), daemon=True)
        self._thread.start()

    @property
    def finished(self) -> bool:
        """The stream has ended (text may still be waiting to be polled)."""
        return self._done.is_set()

    def poll(self) -> list[str]:
        """Every piece received since the last call, without waiting."""
        pieces = []
        while True:
            try:
                pieces.append(self._pieces.get_nowait())
            except queue.Empty:
                return pieces

    def wait_for_pieces(self, timeout: float) -> list[str]:
        """Wait up to ``timeout`` for at least one piece (or the end), then poll."""
        try:
            first = self._pieces.get(timeout=timeout)
        except queue.Empty:
            return []
        return [first, *self.poll()]

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def raise_error(self) -> None:
        """Re-raise, on the caller's thread, an error the stream raised."""
        if self._error is not None:
            raise self._error

    def _run(self, stream: Iterable[str]) -> None:
        try:
            for piece in stream:
                self._pieces.put(piece)
        except Exception as exc:  # handed to the page via raise_error()
            self._error = exc
        finally:
            self._done.set()
