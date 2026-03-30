from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def similarity_score(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def extract_markdown_tables(text: str) -> List[dict]:
    tables = []
    current = []
    for line in text.splitlines():
        if "|" in line:
            current.append(line)
        elif current:
            tables.append({"format": "markdown", "content": "\n".join(current)})
            current = []
    if current:
        tables.append({"format": "markdown", "content": "\n".join(current)})
    return tables


def collect_non_empty(chunks: Iterable[str]) -> List[str]:
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
