"""Shared fixtures for llming-models tests."""
import os
import dotenv

# Load .env from the SalesBot project root (three levels up from tests/)
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
