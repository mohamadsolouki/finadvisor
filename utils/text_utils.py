"""Utility helpers for formatting text for display."""

from __future__ import annotations

import re
from typing import Iterable

# Characters that do not require additional spacing around emphasis markers
_PREFIX_SKIP: Iterable[str] = "([{/'\"“”‘’—-"
_SUFFIX_SKIP: Iterable[str] = ".,;:!?)]}/'\"“”‘’—-"


def normalize_markdown_spacing(text: str | None) -> str | None:
    """Ensure Markdown emphasis markers don't collapse surrounding spaces.

    OpenAI responses sometimes omit whitespace around `*italic*` or `**bold**`
    segments (e.g., ``and*December*``). Streamlit then renders the words
    without visible gaps. This helper adds a thin whitespace buffer around
    emphasis tokens while avoiding double spaces or affecting bullet markers.

    Args:
        text: Raw Markdown text. ``None`` is returned unchanged.

    Returns:
        Cleaned Markdown string with normalized spacing.
    """
    if not text:
        return text

    # Normalize exotic whitespace (non-breaking, thin spaces) to regular spaces
    original = re.sub(r"[\u00a0\u1680\u180e\u2000-\u200b\u202f\u205f\u3000]", " ", text)

    pattern = re.compile(r"(\*{1,2}[^*]+\*{1,2})|(_{1,2}[^_]+_{1,2})")
    pieces: list[str] = []
    last_index = 0

    for match in pattern.finditer(original):
        start, end = match.span()
        segment = match.group(0)
        # Append text preceding the emphasis block unchanged
        pieces.append(original[last_index:start])

        prefix_char = original[start - 1] if start > 0 else ""
        suffix_char = original[end] if end < len(original) else ""

        prefix_space = ""
        suffix_space = ""

        if start > 0 and not prefix_char.isspace() and prefix_char not in _PREFIX_SKIP:
            prefix_space = " "

        if end < len(original) and not suffix_char.isspace() and suffix_char not in _SUFFIX_SKIP:
            suffix_space = " "

        pieces.append(f"{prefix_space}{segment}{suffix_space}")
        last_index = end

    pieces.append(original[last_index:])
    cleaned = "".join(pieces)

    # Collapse any runs of more than two spaces that may have been introduced
    cleaned = re.sub(r"(?<!\S) {2,}", " ", cleaned)

    # Repair accidentally separated emphasis markers like "* *" or "_ _"
    cleaned = re.sub(r"\*\s+\*", "**", cleaned)
    cleaned = re.sub(r"_\s+_", "__", cleaned)

    return cleaned