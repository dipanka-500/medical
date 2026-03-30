"""Cache layer — Redis-backed with graceful degradation."""

from .service import CacheService

__all__ = ["CacheService"]
