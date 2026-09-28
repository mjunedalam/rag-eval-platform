"""The compact status bar under the title: service pills and this session's KPIs."""

import html
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Pill:
    label: str
    ok: bool


def header_html(pills: Sequence[Pill], kpis: Sequence[str]) -> str:
    """Green or red service pills on the left, session numbers on the right."""
    left = "".join(
        f'<span class="rag-pill {"ok" if pill.ok else "down"}">{html.escape(pill.label)}</span>'
        for pill in pills
    )
    right = "".join(f'<span class="rag-kpi">{html.escape(kpi)}</span>' for kpi in kpis)
    return f'<div class="rag-header"><div>{left}</div><div>{right}</div></div>'


# One accent per phase: the panel titles and the health cards use them.
PHASE_ACCENTS = {
    "health": "#d97757",
    "ingest": "#64748b",
    "embed": "#7c3aed",
    "retrieve": "#2563eb",
    "generate": "#d97757",
    "evaluate": "#16a34a",
    "timing": "#0891b2",
}


def panel_title_html(title: str, phase: str) -> str:
    accent = PHASE_ACCENTS.get(phase, "#64748b")
    return f'<div class="rag-panel-title" style="--accent:{accent}">{html.escape(title)}</div>'


# The whole app's extra styling, on top of the theme in .streamlit/config.toml.
APP_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
.block-container {padding-top: 4rem; padding-bottom: 0.5rem;}
h1 {font-size: 1.7rem !important; font-weight: 700 !important; letter-spacing: -0.02em;
  padding: 0 !important;}
h1::before {content: "◆ "; color: #d97757;}
.rag-header {display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
  flex-wrap: wrap;}
.rag-pill, .rag-kpi {display: inline-block; font-size: 0.75rem; border-radius: 999px;
  padding: 0.15rem 0.65rem; margin: 0.1rem 0.3rem 0.1rem 0; border: 1px solid var(--rag-border);
  background: var(--rag-surface);}
.rag-pill::before {content: "●"; margin-right: 0.35rem;}
.rag-pill.ok::before {color: #16a34a;}
.rag-pill.down {border-color: var(--rag-down-border); background: var(--rag-down-bg);}
.rag-pill.down::before {color: #dc2626;}
.rag-kpi {background: var(--rag-soft); border-color: transparent; font-weight: 600;
  color: var(--rag-ink-2);}
[data-testid="stVerticalBlockBorderWrapper"] {background: var(--rag-surface);
  box-shadow: 0 1px 2px var(--rag-shadow), 0 4px 14px var(--rag-shadow);}
.rag-panel-title {font-weight: 650; font-size: 0.92rem; padding-left: 0.55rem;
  border-left: 3px solid var(--accent); margin-bottom: 0.15rem; color: var(--rag-ink);}
.rag-health {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.4rem;}
.rag-health-card {border: 1px solid var(--rag-grid); border-radius: 0.55rem;
  padding: 0.25rem 0.5rem;
  background: var(--rag-surface-2);}
.rag-health-card .v {font-weight: 700; font-size: 0.95rem; color: var(--rag-ink);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.rag-health-card .l {font-size: 0.68rem; color: var(--rag-muted); text-transform: uppercase;
  letter-spacing: 0.04em;}
.rag-health-card .d {display: none;}
.rag-health-card .bar {height: 4px; background: var(--rag-grid); border-radius: 2px;
  margin-top: 3px;}
.rag-health-card .bar span {display: block; height: 100%; border-radius: 2px;}
.rag-health-card.good .bar span {background: #16a34a;}
.rag-health-card.fair .bar span {background: #d97706;}
.rag-health-card.poor .bar span {background: #dc2626;}
.st-key-ov-answer [data-testid="stMarkdownContainer"] :is(p, li, td, th) {font-size: 0.85rem;}
.st-key-ov-answer [data-testid="stMarkdownContainer"] :is(h1, h2, h3) {font-size: 0.95rem;
  padding: 0.3rem 0 0.1rem;}
</style>"""


# The app's own colours for each Streamlit theme; the CSS above only uses these variables.
_LIGHT = {
    "page": "#faf9f5",
    "surface": "#ffffff",
    "surface-2": "#fcfbf8",
    "soft": "#f0eee6",
    "tile": "#f9fafb",
    "border": "#e5e2d9",
    "border-strong": "#d6d3c9",
    "grid": "#ece9e1",
    "ink": "#1f1e1d",
    "ink-2": "#3d3929",
    "muted": "#6b7280",
    "shimmer-hi": "#111827",
    "shadow": "rgba(31, 30, 29, 0.05)",
    "down-bg": "#fef2f2",
    "down-border": "#fecaca",
}
_DARK = {
    "page": "#1f1e1d",
    "surface": "#262624",
    "surface-2": "#2c2b28",
    "soft": "#383733",
    "tile": "#2c2b28",
    "border": "#3d3c38",
    "border-strong": "#4a4945",
    "grid": "#3a3935",
    "ink": "#f5f4ef",
    "ink-2": "#e0ddd4",
    "muted": "#a3a097",
    "shimmer-hi": "#faf9f5",
    "shadow": "rgba(0, 0, 0, 0.35)",
    "down-bg": "#3b1f1f",
    "down-border": "#7f1d1d",
}


def theme_css() -> str:
    """The ``:root`` rule: every colour as ``light-dark(light, dark)``.

    Streamlit sets ``color-scheme`` on the app to ``light`` or ``dark`` (⋮ → Settings →
    Theme), and the browser resolves ``light-dark()`` where each colour is used. So the page
    follows a theme switch instantly, without a rerun.
    """
    body = " ".join(f"--rag-{name}: light-dark({_LIGHT[name]}, {_DARK[name]});"
                    for name in _LIGHT)  # fmt: skip
    return f":root {{{body}}}"


def app_css() -> str:
    """APP_CSS with the theme's colours, as one style block (``@import`` must come first)."""
    style, imports, rest = APP_CSS.split("\n", 2)
    return "\n".join([style, imports, theme_css(), rest])
