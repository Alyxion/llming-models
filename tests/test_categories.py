"""Tests for llming_models.model_categories — ModelCategories."""

from llming_models.model_categories import ModelCategories


class TestModelCategories:
    def test_constants(self):
        assert ModelCategories.SMALL == "small"
        assert ModelCategories.MEDIUM == "medium"
        assert ModelCategories.LARGE == "large"
        assert ModelCategories.REASONING_SMALL == "reasoning_small"
        assert ModelCategories.REASONING_MEDIUM == "reasoning_medium"
        assert ModelCategories.REASONING_LARGE == "reasoning_large"

    def test_all_categories_in_set(self):
        assert "small" in ModelCategories.MODEL_CATEGORIES
        assert "large" in ModelCategories.MODEL_CATEGORIES
        assert "reasoning_large" in ModelCategories.MODEL_CATEGORIES
        assert len(ModelCategories.MODEL_CATEGORIES) == 6
