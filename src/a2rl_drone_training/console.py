from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


ConsoleSection = tuple[str, Sequence[tuple[str, Any]]]


def format_table(
    sections: Iterable[ConsoleSection],
    *,
    max_value_width: int = 88,
) -> str:
    """Render metrics in the compact, grouped table style used by SB3."""

    rows: list[tuple[str, str]] = []
    for section, metrics in sections:
        rows.append((f"{section}/", ""))
        rows.extend((f"    {name}", _stringify(value)) for name, value in metrics)

    if not rows:
        return ""

    key_width = max(len(key) for key, _ in rows)
    value_width = min(
        max((len(value) for _, value in rows), default=0),
        max_value_width,
    )
    value_width = max(value_width, 1)
    border = "-" * (key_width + value_width + 7)
    lines = [border]
    for key, value in rows:
        lines.append(
            f"| {key:<{key_width}} | {_ellipsize(value, value_width):<{value_width}} |"
        )
    lines.append(border)
    return "\n".join(lines)


def print_table(sections: Iterable[ConsoleSection]) -> None:
    print(format_table(sections), flush=True)


def _stringify(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _ellipsize(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."
