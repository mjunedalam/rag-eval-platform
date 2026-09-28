"""Pure pieces of the live Overview: status wording for each pipeline step, smooth-typing
chunks, the stats tiles, and the small HTML/CSS for the shimmering status and the cursor."""

import html
from dataclasses import dataclass

from rag_eval_platform.playground.core import QueryTrace

PENDING = "…"  # shown in a tile until its value is known

_STATUS = {
    "embed": "Understanding your question…",
    "search": "Searching {chunks:,} chunks…",
    "rerank": "Re-ranking {candidates} candidates…",
    "read": "Reading {sources} sources…",
    "think": "Thinking…",
    "write": "Writing the answer…",
    "cite": "Checking citations…",
    "done": "Done in {seconds:.1f} s",
}

# A shimmer that sweeps across the status text (like a model "working"), and a blinking cursor.
LIVE_CSS = """<style>
@keyframes rag-shimmer {0% {background-position: 200% 0} 100% {background-position: -200% 0}}
.rag-status {font-size: 0.85rem; font-weight: 500; display: inline-block;
  background: linear-gradient(90deg, #9ca3af 30%, var(--rag-shimmer-hi) 50%, #9ca3af 70%);
  background-size: 200% 100%; -webkit-background-clip: text; background-clip: text;
  color: transparent; animation: rag-shimmer 1.8s linear infinite;}
.rag-status.done {animation: none; background: none; color: #16a34a;}
@keyframes rag-spin {to {transform: rotate(360deg)}}
.rag-spark {display: inline-block; color: #d97757; font-weight: 700;
  animation: rag-spin 2.4s linear infinite;}
@keyframes rag-breathe {0%, 100% {opacity: 0.25; transform: scale(0.8)} 50% {opacity: 1;
  transform: scale(1)}}
.rag-cursor {display: inline-block; margin-left: 3px; color: #d97757; font-size: 0.75em;
  animation: rag-breathe 1s ease-in-out infinite;}
@keyframes rag-fade {from {opacity: 0; filter: blur(1.5px)} to {opacity: 1; filter: none}}
.rag-timeline {position: relative; margin: 0.1rem 0 0.5rem 0.25rem; padding-left: 0.95rem;
  border-left: 1.5px solid var(--rag-border); font-size: 0.8rem; color: var(--rag-muted);
  line-height: 1.6;}
.rag-step {position: relative;}
.rag-step::before {content: ""; position: absolute; left: calc(-0.95rem - 4.25px); top: 0.6em;
  width: 7px; height: 7px; border-radius: 50%; background: #16a34a;
  box-shadow: 0 0 0 2px var(--rag-page);}
.rag-step:last-child {animation: rag-fade 0.4s ease-out;}
.rag-tiles {display: flex; gap: 0.5rem; flex-wrap: nowrap; margin: 0.2rem 0 0.4rem 0;}
.rag-tile {flex: 1; min-width: 0; border: 1px solid var(--rag-border); border-radius: 8px;
  padding: 0.25rem 0.5rem; background: var(--rag-tile);}
.rag-tile .label {font-size: 0.68rem; color: var(--rag-muted); white-space: nowrap;}
.rag-tile .value {font-size: 1.05rem; font-weight: 700; color: var(--rag-shimmer-hi);
  white-space: nowrap;}
@keyframes rag-glow {0%, 100% {filter: drop-shadow(0 0 0 rgba(217,119,6,0));}
  50% {filter: drop-shadow(0 0 7px rgba(217,119,6,0.9));}}
g.node.active {animation: rag-glow 1.4s ease-in-out infinite;}
@keyframes rag-march {to {stroke-dashoffset: -20;}}
g.edge.flow path {stroke: #d97706 !important; stroke-width: 2px; stroke-dasharray: 6 4;
  animation: rag-march 0.7s linear infinite;}
g.edge.flow polygon {fill: #d97706 !important; stroke: #d97706 !important;}
</style>"""


def status_text(
    step: str, *, chunks: int = 0, candidates: int = 0, sources: int = 0, seconds: float = 0.0
) -> str:
    """What the pipeline is doing right now, in plain words."""
    if step not in _STATUS:
        raise ValueError(f"unknown step: {step}")
    return _STATUS[step].format(
        chunks=chunks, candidates=candidates, sources=sources, seconds=seconds
    )


@dataclass(frozen=True)
class Pace:
    """How fast the live run plays: normal, or slow motion for watching each step."""

    step_pause: float  # seconds a status stays up between phases
    text_frame_s: float  # seconds between updates of the text being written
    min_cps: float  # the slowest the text is revealed, in characters per second
    catch_up_s: float | None  # empty a backlog within this; None = fixed pace (slow motion)
    verb_every_s: float  # how often the status word changes
    fade_s: float  # how long new words take to go from faint and blurred to sharp
    frames: int  # frames when a chart grows in
    frame_seconds: float  # seconds per chart frame


def pace(*, slow: bool) -> Pace:
    if slow:
        return Pace(step_pause=1.6, text_frame_s=0.06, min_cps=18, catch_up_s=None,
                    verb_every_s=3.0, fade_s=1.6, frames=24, frame_seconds=0.06)  # fmt: skip
    # Normal speed keeps up with the model: a backlog is revealed within 0.3 s, steadily.
    return Pace(step_pause=0.35, text_frame_s=0.04, min_cps=60, catch_up_s=0.3,
                verb_every_s=1.5, fade_s=0.9, frames=8, frame_seconds=0.05)  # fmt: skip


def tail_for_display(text: str, limit: int = 420) -> str:
    """The newest part of a growing answer, starting at a word, so the cursor stays in view."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    space = tail.find(" ")
    return "…" + (tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail)


@dataclass(frozen=True)
class LiveStats:
    chunks: int | None = None
    candidates: int | None = None
    top_similarity: float | None = None
    sources: int | None = None
    tokens: int | None = None
    generate_s: float | None = None  # from the request to the last token
    first_token_s: float | None = None  # waiting before the first token (model load, prompt)
    cited: int | None = None
    elapsed_s: float | None = None

    @property
    def tokens_per_s(self) -> float | None:
        """Writing speed, measured from the first token when known (so model loading is not
        counted as slow writing)."""
        if self.tokens is None or self.generate_s is None:
            return None
        writing = self.generate_s - (self.first_token_s or 0.0)
        return self.tokens / writing if writing > 0 else None


def stat_tiles(stats: LiveStats) -> tuple[tuple[str, str], ...]:
    """(label, value) for each tile; values not known yet show a placeholder."""

    def shown(value: object, fmt: str) -> str:
        return PENDING if value is None else fmt.format(value)

    cited = (
        PENDING if stats.cited is None or stats.sources is None
        else f"{stats.cited}/{stats.sources}"
    )  # fmt: skip
    return (
        ("chunks searched", shown(stats.chunks, "{:,}")),
        ("candidates", shown(stats.candidates, "{}")),
        ("top similarity", shown(stats.top_similarity, "{:.2f}")),
        ("sources sent", shown(stats.sources, "{}")),
        ("first token", shown(stats.first_token_s, "{:.1f} s")),
        ("tokens", shown(stats.tokens, "{}")),
        ("speed", shown(stats.tokens_per_s, "{:.0f} tok/s")),
        ("cited", cited),
        ("elapsed", shown(stats.elapsed_s, "{:.1f} s")),
    )


def stats_from_trace(
    trace: QueryTrace, chunks: int | None, first_token_s: float | None = None
) -> LiveStats:
    """The finished run's stats, for redrawing the Overview after the live run."""
    answer = trace.answer
    return LiveStats(
        chunks=chunks,
        candidates=len(trace.vector_results),
        top_similarity=trace.vector_results[0].score if trace.vector_results else None,
        sources=len(trace.final_results),
        tokens=answer.output_tokens,
        generate_s=answer.latency_ms / 1000,
        first_token_s=first_token_s,
        cited=len({c.number for c in answer.citations}),
        elapsed_s=(trace.retrieval_ms + answer.latency_ms) / 1000,
    )


def status_html(text: str, *, done: bool = False) -> str:
    """The status line; while running it gets Claude's spinning spark and a shimmer."""
    if done:
        return f'<span class="rag-status done">{html.escape(text)}</span>'
    return f'<span class="rag-spark">✻</span> <span class="rag-status">{html.escape(text)}</span>'


def typing_html(text: str, *, cursor: bool = True) -> str:
    body = html.escape(text).replace("\n", "<br>")
    return body + ('<span class="rag-cursor">●</span>' if cursor else "")


def tiles_html(tiles: tuple[tuple[str, str], ...]) -> str:
    cells = "".join(
        f'<div class="rag-tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in tiles
    )
    return f'<div class="rag-tiles">{cells}</div>'
