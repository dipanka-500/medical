from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from medicscan_ocr.config import Settings
from medicscan_ocr.schemas import AnalysisResult, BackendResult, Section


class OCRBackend(ABC):
    name = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def availability(self) -> tuple[bool, str | None]:
        """Return whether this backend can run in the current environment."""
        return True, None

    @abstractmethod
    def run(
        self,
        input_path: str,
        analysis: AnalysisResult,
        output_dir: Optional[str] = None,
    ) -> BackendResult:
        raise NotImplementedError


class PlaceholderBackend(OCRBackend):
    name = "placeholder"

    def __init__(self, settings: Settings, name: str, message: str, docs_url: str = "") -> None:
        super().__init__(settings)
        self.name = name
        self.message = message
        self.docs_url = docs_url

    def availability(self) -> tuple[bool, str | None]:
        return False, "placeholder"

    def run(
        self,
        input_path: str,
        analysis: AnalysisResult,
        output_dir: Optional[str] = None,
    ) -> BackendResult:
        return BackendResult(
            backend=self.name,
            status="skipped",
            confidence=0.0,
            metadata={
                "message": self.message,
                "docs_url": self.docs_url,
                "input_path": str(Path(input_path).resolve()),
            },
        )


class BackendRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backends: Dict[str, OCRBackend] = {}

    def register(self, backend: OCRBackend) -> None:
        self._backends[backend.name] = backend

    def get(self, name: str) -> OCRBackend:
        if name not in self._backends:
            raise KeyError("Backend not registered: {0}".format(name))
        return self._backends[name]

    def has(self, name: str) -> bool:
        return name in self._backends

    def availability(self, name: str) -> tuple[bool, str | None]:
        return self.get(name).availability()

    def status(self, name: str) -> dict[str, object]:
        backend = self.get(name)
        available, detail = backend.availability()
        mode = "placeholder" if detail == "placeholder" else ("ready" if available else "unavailable")
        return {
            "registered": True,
            "available": available,
            "mode": mode,
            "detail": detail,
            "backend_class": backend.__class__.__name__,
        }

    def all_statuses(self) -> dict[str, dict[str, object]]:
        return {name: self.status(name) for name in sorted(self._backends)}
