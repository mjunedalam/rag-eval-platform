"""Tests for playground.header: the compact status bar under the title."""

import re

from rag_eval_platform.playground.header import (
    APP_CSS,
    PHASE_ACCENTS,
    Pill,
    app_css,
    header_html,
    panel_title_html,
    theme_css,
)
from rag_eval_platform.playground.live import LIVE_CSS


def test_header_shows_service_pills_and_session_kpis() -> None:
    html = header_html(
        (Pill("Chroma · 11 chunks", ok=True), Pill("Ollama down", ok=False)),
        ("3 questions", "avg 3.3 s"),
    )

    assert html.startswith('<div class="rag-header">')
    assert 'class="rag-pill ok"' in html
    assert 'class="rag-pill down"' in html
    assert html.count('class="rag-kpi"') == 2
    assert "3 questions" in html


def test_header_escapes_text_and_hides_empty_kpis() -> None:
    html = header_html((Pill("<b>x</b>", ok=True),), ())

    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "rag-kpi" not in html


def test_panel_titles_carry_their_phase_colour() -> None:
    html = panel_title_html("③ Retrieve · similarity", "retrieve")

    assert html.startswith('<div class="rag-panel-title"')
    assert PHASE_ACCENTS["retrieve"] in html
    assert "③ Retrieve · similarity" in html
    assert panel_title_html("<x>", "timing").count("&lt;x&gt;") == 1


def test_app_css_loads_the_serif_font_and_styles_every_new_piece() -> None:
    for selector in (".rag-header", ".rag-pill", ".rag-health-card", ".rag-panel-title"):
        assert selector in APP_CSS
    assert "Source+Serif" in APP_CSS


def _defined(css: str) -> dict[str, str]:
    return dict(re.findall(r"(--rag-[\w-]+):\s*([^;]+);", css))


def test_every_colour_has_a_light_and_a_dark_value() -> None:
    colours = _defined(theme_css())

    for name, value in colours.items():
        pair = re.fullmatch(r"light-dark\((.+), (.+)\)", value)
        assert pair, name
        if name not in {"--rag-accent"}:
            assert pair.group(1) != pair.group(2), name  # e.g. dark ink on light, light on dark
    assert {"--rag-ink", "--rag-surface", "--rag-muted"} <= set(colours)


def test_every_colour_the_css_uses_is_defined_by_the_theme() -> None:
    used = set(re.findall(r"var\((--rag-[\w-]+)\)", APP_CSS + LIVE_CSS))

    assert used
    assert used <= set(_defined(theme_css()))


def test_app_css_is_one_style_block_with_import_first() -> None:
    css = app_css()

    assert css.startswith("<style>")
    assert css.count("<style>") == 1  # two blocks in one Markdown call print as text
    assert css.index("@import") < css.index(":root")  # @import is ignored after any rule
    assert _defined(theme_css()).items() <= _defined(css).items()
