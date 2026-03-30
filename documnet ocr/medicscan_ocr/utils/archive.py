from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_extract_zip(
    archive_path: str | Path,
    target_dir: str | Path,
    max_members: int = 2048,
    max_total_size_bytes: int = 512 * 1024 * 1024,
) -> Path:
    archive_path = Path(archive_path).resolve()
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(archive_path), "r") as archive:
        members = archive.infolist()
        if len(members) > max_members:
            raise ValueError(
                "ZIP archive contains too many members ({0}, max {1}).".format(
                    len(members), max_members
                )
            )

        total_size = 0
        for member in members:
            if member.is_dir():
                continue

            total_size += int(member.file_size)
            if total_size > max_total_size_bytes:
                raise ValueError("ZIP archive exceeds the maximum allowed extracted size.")

            destination = (target_dir / member.filename).resolve()
            if not destination.is_relative_to(target_dir):
                raise ValueError(
                    "ZIP archive contains an unsafe path: {0}".format(member.filename)
                )

            if member.filename != member.filename.replace("\\", "/"):
                raise ValueError(
                    "ZIP archive contains a suspicious backslash path: {0}".format(
                        member.filename
                    )
                )

        for member in members:
            destination = (target_dir / member.filename).resolve()
            if not destination.is_relative_to(target_dir):
                raise ValueError(
                    "ZIP archive contains an unsafe path: {0}".format(member.filename)
                )

            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, open(destination, "wb") as target:
                    while True:
                        chunk = source.read(1024 * 64)
                        if not chunk:
                            break
                        target.write(chunk)

        logger.debug(
            "Extracted %d members from %s to %s", len(members), archive_path, target_dir
        )
    return target_dir
