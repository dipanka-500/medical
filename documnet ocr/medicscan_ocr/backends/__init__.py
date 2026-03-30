from __future__ import annotations

from medicscan_ocr.backends.base import BackendRegistry, PlaceholderBackend
from medicscan_ocr.backends.firered import FireRedBackend
from medicscan_ocr.backends.granite_vision import GraniteVisionBackend
from medicscan_ocr.backends.local_commands import ChandraCommandBackend, SuryaCommandBackend
from medicscan_ocr.backends.sarvam import SarvamVisionBackend


def build_registry(settings):
    registry = BackendRegistry(settings)
    registry.register(SarvamVisionBackend(settings))
    registry.register(SuryaCommandBackend(settings))
    registry.register(ChandraCommandBackend(settings))
    registry.register(FireRedBackend(settings))
    registry.register(GraniteVisionBackend(settings))
    registry.register(
        PlaceholderBackend(
            settings,
            name="doctr_placeholder",
            message="docTR is wired as a lightweight fallback but not fully implemented in this fresh scaffold.",
            docs_url="https://github.com/mindee/doctr",
        )
    )
    registry.register(
        PlaceholderBackend(
            settings,
            name="donut_placeholder",
            message="Donut is included in routing as a structure-aware reviewer. Add a transformers-based backend to enable execution.",
            docs_url="https://huggingface.co/docs/transformers/main/model_doc/donut",
        )
    )
    registry.register(
        PlaceholderBackend(
            settings,
            name="pixtral_placeholder",
            message="Pixtral is included as a multimodal review backend for difficult layouts. Add a transformers or API adapter to enable execution.",
            docs_url="https://huggingface.co/docs/transformers/main/model_doc/pixtral",
        )
    )
    registry.register(
        PlaceholderBackend(
            settings,
            name="layoutlmv3_placeholder",
            message="LayoutLMv3 is tracked as a layout enricher. Add a transformers-based implementation if you want section-level layout predictions.",
            docs_url="https://huggingface.co/docs/transformers/main/model_doc/layoutlmv3",
        )
    )
    registry.register(
        PlaceholderBackend(
            settings,
            name="table_transformer_placeholder",
            message="Table Transformer is tracked as a table enricher. Add a transformers-based implementation to enable cell extraction.",
            docs_url="https://huggingface.co/docs/transformers/main/model_doc/table-transformer",
        )
    )
    return registry
