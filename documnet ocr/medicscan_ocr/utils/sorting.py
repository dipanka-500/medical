from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List


_NUMBER_RE = re.compile(r"(\d+)")


def natural_sort_key(value: Any):
    text = str(value)
    parts = _NUMBER_RE.split(text.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def natural_sorted_paths(paths: Iterable[Path]) -> List[Path]:
    return sorted(paths, key=lambda path: natural_sort_key(path.name))

