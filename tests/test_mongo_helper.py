"""Tests for budget/_mongo.py -- shared async MongoDB client helper.

Fully mocks pymongo.AsyncMongoClient to test caching, URL resolution,
and error handling without any real database connection.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

# We need to reset the module-level cache between tests, so we import the
# module itself and manipulate _async_cache directly.
import llming_models.budget._mongo as mongo_mod
from llming_models.budget._mongo import get_async_mongo_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the module-level client cache before each test."""
    mongo_mod._async_cache.clear()


# ---------------------------------------------------------------------------
# Basic client creation
# ---------------------------------------------------------------------------

class TestGetAsyncMongoClient:
    """get_async_mongo_client creates and caches AsyncMongoClient instances."""

    @patch("pymongo.AsyncMongoClient")
    def test_returns_client_for_url(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        result = get_async_mongo_client("mongodb://test:27017")
        assert result is mock_client
        mock_cls.assert_called_once()

    @patch("pymongo.AsyncMongoClient")
    def test_same_url_returns_cached_client(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        client1 = get_async_mongo_client("mongodb://test:27017")
        client2 = get_async_mongo_client("mongodb://test:27017")
        assert client1 is client2
        # Only one AsyncMongoClient instance should have been created
        assert mock_cls.call_count == 1

    @patch("pymongo.AsyncMongoClient")
    def test_different_url_returns_different_client(self, mock_cls: MagicMock) -> None:
        mock_cls.side_effect = [MagicMock(name="client-a"), MagicMock(name="client-b")]
        client1 = get_async_mongo_client("mongodb://host-a:27017")
        client2 = get_async_mongo_client("mongodb://host-b:27017")
        assert client1 is not client2
        assert mock_cls.call_count == 2


# ---------------------------------------------------------------------------
# URL from environment variables
# ---------------------------------------------------------------------------

class TestEnvVarResolution:
    """URL resolution from environment variables when no explicit URL is given."""

    @patch("pymongo.AsyncMongoClient")
    @patch.dict("os.environ", {"MONGODB_CONNECTION": "mongodb://from-env:27017"}, clear=False)
    def test_url_from_mongodb_connection_env(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        result = get_async_mongo_client()
        assert result is mock_client
        # Should have used the env var URL
        call_args = mock_cls.call_args[0]
        assert call_args[0] == "mongodb://from-env:27017"

    @patch("pymongo.AsyncMongoClient")
    @patch.dict("os.environ", {"O365_MONGODB_URL": "mongodb://fallback:27017"}, clear=False)
    def test_url_from_o365_env_fallback(self, mock_cls: MagicMock) -> None:
        # Remove MONGODB_CONNECTION if present
        import os
        env_backup = os.environ.pop("MONGODB_CONNECTION", None)
        try:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            result = get_async_mongo_client()
            assert result is mock_client
            call_args = mock_cls.call_args[0]
            assert call_args[0] == "mongodb://fallback:27017"
        finally:
            if env_backup is not None:
                os.environ["MONGODB_CONNECTION"] = env_backup

    @patch("pymongo.AsyncMongoClient")
    @patch.dict("os.environ", {
        "MONGODB_CONNECTION": "mongodb://primary:27017",
        "O365_MONGODB_URL": "mongodb://secondary:27017",
    }, clear=False)
    def test_mongodb_connection_takes_precedence(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        result = get_async_mongo_client()
        call_args = mock_cls.call_args[0]
        assert call_args[0] == "mongodb://primary:27017"


# ---------------------------------------------------------------------------
# Missing URL
# ---------------------------------------------------------------------------

class TestMissingUrl:
    """Missing URL raises ValueError."""

    @patch.dict("os.environ", {}, clear=True)
    def test_no_url_no_env_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="MongoDB URL must be provided"):
            get_async_mongo_client()

    @patch.dict("os.environ", {}, clear=True)
    def test_none_url_no_env_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="MongoDB URL must be provided"):
            get_async_mongo_client(None)

    @patch.dict("os.environ", {}, clear=True)
    def test_empty_string_url_no_env_raises_value_error(self) -> None:
        """Empty string is falsy so should fall through to env lookup, then fail."""
        with pytest.raises(ValueError, match="MongoDB URL must be provided"):
            get_async_mongo_client("")


# ---------------------------------------------------------------------------
# Thread safety -- cache is protected by lock
# ---------------------------------------------------------------------------

class TestCacheLock:
    """Verify the lock is a threading.Lock."""

    def test_lock_exists(self) -> None:
        lock = mongo_mod._async_lock
        assert hasattr(lock, "acquire") and hasattr(lock, "release")


# ---------------------------------------------------------------------------
# Client constructor kwargs
# ---------------------------------------------------------------------------

class TestClientConstructorKwargs:
    """Verify the expected connection kwargs are passed."""

    @patch("pymongo.AsyncMongoClient")
    def test_passes_expected_kwargs(self, mock_cls: MagicMock) -> None:
        get_async_mongo_client("mongodb://test:27017")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["serverSelectionTimeoutMS"] == 15000
        assert call_kwargs["connectTimeoutMS"] == 10000
        assert call_kwargs["maxPoolSize"] == 10

    @patch("pymongo.AsyncMongoClient")
    def test_url_is_first_positional_arg(self, mock_cls: MagicMock) -> None:
        get_async_mongo_client("mongodb://myhost:27017")
        call_args = mock_cls.call_args[0]
        assert call_args[0] == "mongodb://myhost:27017"
