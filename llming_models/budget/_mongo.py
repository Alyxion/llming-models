"""Shared async MongoDB client helper."""

import os
import threading

_async_cache: dict = {}
_async_lock = threading.Lock()


def get_async_mongo_client(url: str | None = None):
    """Return a shared AsyncMongoClient for the given URL."""
    from pymongo import AsyncMongoClient

    url = url or os.getenv("MONGODB_CONNECTION") or os.getenv("O365_MONGODB_URL")
    if not url:
        raise ValueError("MongoDB URL must be provided or set in MONGODB_CONNECTION env var")
    with _async_lock:
        if url not in _async_cache:
            _async_cache[url] = AsyncMongoClient(
                url,
                serverSelectionTimeoutMS=15000,
                connectTimeoutMS=10000,
                maxPoolSize=10,
            )
        return _async_cache[url]
