from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentType(str, Enum):
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LayoutComplexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class PreprocessResult:
    input_path: str
    processed_path: str
    was_modified: bool
    applied_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedPage:
    page_number: int
    source_name: str
    original_path: str
    processed_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedDocument:
    input_path: str
    source_type: str
    native_path: str
    backend_input_path: str
    pages: List[PreparedPage] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    document_type: DocumentType
    handwritten_confidence: float
    language_code: str
    language_family: str
    language_confidence: float
    layout_complexity: LayoutComplexity
    needs_table_model: bool
    needs_formula_model: bool
    needs_layout_model: bool
    source_hints: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingDecision:
    primary_backend: str
    secondary_backends: List[str]
    enrichers: List[str]
    reason: List[str]
    requested_backend: Optional[str] = None


@dataclass(frozen=True)
class Section:
    type: str
    text: str = ""
    data: Any = None
    confidence: float = 0.0
    bbox: Optional[List[int]] = None


@dataclass(frozen=True)
class BackendResult:
    backend: str
    status: str
    raw_text: str = ""
    sections: List[Section] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(frozen=True)
class OCRResult:
    input_path: str
    preprocessed_path: str
    raw_text: str
    structured: List[Section]
    language: str
    document_type: str
    confidence: float
    tables: List[Dict[str, Any]]
    handwritten_detected: bool
    uncertain_regions: List[str]
    route: RoutingDecision
    analysis: AnalysisResult
    backend_results: List[BackendResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return serialize(self)


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: serialize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value
