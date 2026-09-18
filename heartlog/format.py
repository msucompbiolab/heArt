"""Line formatting for the canonical Advanced/Raw Log record.

The layout is fixed and shared with the service node, which renders this repository's stdout
as the Raw log inside its ``Advanced`` panel::

    HH:MM:SS.mmm  LEVEL  source/stage   message  key=value key=value

Plain text with a ``key=value`` tail -- deliberately not JSON, so a record stays greppable and
copy-pasteable into a bug report. Verbatim subprocess output is emitted UNPREFIXED; the absence
of a prefix is what marks a line as passthrough.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

# The only three levels. There is deliberately no DEBUG: anything worth keeping is INFO, and
# anything not worth keeping does not belong on a stream a human watches live.
LEVELS = ("INFO", "WARN", "ERROR")
_LEVEL_RANK = {name: i for i, name in enumerate(LEVELS)}


def level_rank(level: str) -> int:
    """Ordinal of ``level`` for threshold comparisons; unknown levels sort as INFO."""
    return _LEVEL_RANK.get(str(level).upper(), 0)


def default_clock() -> float:
    return time.time()


# Severity markers the pre-standard call sites embed at the head of the message text
# ("WARNING: ...", "ERROR ...", "NOTE ..."). Ordered longest-first so WARNING wins over WARN.
_SEVERITY_MARKERS = (
    ("ERROR", "ERROR"),
    ("WARNING", "WARN"),
    ("WARN", "WARN"),
    ("NOTE", "INFO"),
)


def infer_level(message: Any, default: str = "INFO") -> str:
    """Infer a record level from a leading severity word in ``message``.

    Severity used to be encoded as a bare word inside the message string rather than as a level
    (WARNING x175, WARN x48, ERROR x42, NOTE x15, two spellings of the same thing coexisting in
    single files). This lifts that word to a real level while the text is emitted VERBATIM --
    the message bodies are a parsing contract for the autonomous loops' failure signatures.
    """
    head = str(message).lstrip().upper()
    for marker, level in _SEVERITY_MARKERS:
        if head.startswith(marker):
            return level
    return default


def format_kv(**kv: Any) -> str:
    """Render ``key=value`` pairs, dropping ``None`` and rounding floats to 3 places.

    Values containing whitespace are quoted so a consumer splitting the tail on spaces cannot
    mistake a message fragment for another pair.
    """
    parts: list[str] = []
    for key, value in kv.items():
        if value is None:
            continue
        if isinstance(value, float):
            rendered = f"{value:.3f}".rstrip("0").rstrip(".")
            # A value that rounds away (-0.0, -1e-9) renders as "-0"; drop the sign so a reader
            # does not read a signed zero as a real negative quantity.
            if rendered == "-0":
                rendered = "0"
        elif isinstance(value, bool):
            rendered = "1" if value else "0"
        else:
            rendered = str(value)
        if rendered == "":
            continue
        if any(c.isspace() for c in rendered):
            rendered = '"%s"' % rendered.replace('"', "'")
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def format_line(
    level: str,
    source: str,
    stage: str,
    message: str,
    /,
    *,
    timestamp: Optional[float] = None,
    **kv: Any,
) -> str:
    """Format one prefixed record. See the module docstring for the layout.

    The leading four are positional-only: a caller passing ``stage=`` or ``level=`` as DETAIL
    (e.g. ``warn("loading", "recovered", stage="ED")``) must land in ``kv`` rather than collide
    with a parameter and raise ``TypeError`` deep inside a solver's recovery path.
    """
    epoch = default_clock() if timestamp is None else timestamp
    stamp = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone()
    ts = f"{stamp:%H:%M:%S}.{stamp.microsecond // 1000:03d}"
    origin = f"{source}/{stage}"
    tail = format_kv(**kv)
    line = f"{ts} {level:<5} {origin:<14} {message}"
    return f"{line}  {tail}" if tail else line
