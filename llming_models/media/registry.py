"""Media provider registry — mirrors providers/__init__.py for LLMs."""
from __future__ import annotations

from typing import Type

from llming_models.media.provider import BaseMediaProvider

MEDIA_PROVIDERS: dict[str, Type[BaseMediaProvider]] = {}


def register_media_provider(name: str):
    """Decorator to register a media provider class."""
    def decorator(cls: Type[BaseMediaProvider]) -> Type[BaseMediaProvider]:
        MEDIA_PROVIDERS[name] = cls
        return cls
    return decorator


def get_media_provider(name: str) -> Type[BaseMediaProvider]:
    """Look up a registered media provider class by name.

    Raises:
        ValueError: If the provider name is not registered.
    """
    if name not in MEDIA_PROVIDERS:
        available = list(MEDIA_PROVIDERS.keys())
        raise ValueError(f"Media provider {name!r} not found (available: {available})")
    return MEDIA_PROVIDERS[name]
