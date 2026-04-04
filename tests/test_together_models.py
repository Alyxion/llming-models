"""Unit tests for Together model configurations."""
from __future__ import annotations

from llming_models.model_info import LLMInfo, ModelSize
from llming_models.providers.together.together_models import (
    BASIC_MODEL,
    TOGETHER_MODELS,
)
from llming_models.providers.together.deepseek.deepseek_models import (
    BASIC_MODEL as DEEPSEEK_BASIC_MODEL,
    TOGETHER_DEEPSEEK_MODELS,
)


# ---------------------------------------------------------------------------
# TOGETHER_MODELS
# ---------------------------------------------------------------------------


class TestTogetherModels:
    """Tests for the TOGETHER_MODELS list."""

    def test_not_empty(self):
        assert len(TOGETHER_MODELS) > 0

    def test_all_are_llm_info(self):
        for model in TOGETHER_MODELS:
            assert isinstance(model, LLMInfo)

    def test_all_have_together_provider(self):
        for model in TOGETHER_MODELS:
            assert model.provider == "together"

    def test_all_have_required_fields(self):
        for model in TOGETHER_MODELS:
            assert model.name
            assert model.label
            assert model.model
            assert model.description
            assert model.input_token_price > 0
            assert model.output_token_price > 0
            assert model.max_input_tokens > 0
            assert model.max_output_tokens > 0

    def test_all_have_api_base(self):
        for model in TOGETHER_MODELS:
            assert model.api_base == "https://api.together.xyz/v1"

    def test_unique_names(self):
        names = [m.name for m in TOGETHER_MODELS]
        assert len(names) == len(set(names))

    def test_unique_model_ids(self):
        models = [m.model for m in TOGETHER_MODELS]
        assert len(models) == len(set(models))


# ---------------------------------------------------------------------------
# BASIC_MODEL
# ---------------------------------------------------------------------------


class TestBasicModel:
    """Tests for BASIC_MODEL selection."""

    def test_is_medium_size(self):
        assert BASIC_MODEL.size == ModelSize.MEDIUM

    def test_is_in_together_models(self):
        assert BASIC_MODEL in TOGETHER_MODELS

    def test_has_valid_fields(self):
        assert BASIC_MODEL.provider == "together"
        assert BASIC_MODEL.name
        assert BASIC_MODEL.model


# ---------------------------------------------------------------------------
# DeepSeek Models
# ---------------------------------------------------------------------------


class TestDeepSeekModels:
    """Tests for TOGETHER_DEEPSEEK_MODELS."""

    def test_contains_two_models(self):
        assert len(TOGETHER_DEEPSEEK_MODELS) == 2

    def test_has_reasoner_and_chat(self):
        names = {m.name for m in TOGETHER_DEEPSEEK_MODELS}
        assert "deepseek_reasoner" in names
        assert "deepseek_chat" in names

    def test_reasoner_is_large(self):
        reasoner = next(m for m in TOGETHER_DEEPSEEK_MODELS if m.name == "deepseek_reasoner")
        assert reasoner.size == ModelSize.LARGE

    def test_chat_is_medium(self):
        chat = next(m for m in TOGETHER_DEEPSEEK_MODELS if m.name == "deepseek_chat")
        assert chat.size == ModelSize.MEDIUM

    def test_deepseek_basic_model_matches(self):
        assert DEEPSEEK_BASIC_MODEL.size == ModelSize.MEDIUM
        assert DEEPSEEK_BASIC_MODEL in TOGETHER_DEEPSEEK_MODELS

    def test_reasoner_has_higher_price(self):
        reasoner = next(m for m in TOGETHER_DEEPSEEK_MODELS if m.name == "deepseek_reasoner")
        chat = next(m for m in TOGETHER_DEEPSEEK_MODELS if m.name == "deepseek_chat")
        assert reasoner.input_token_price > chat.input_token_price
        assert reasoner.output_token_price > chat.output_token_price

    def test_together_models_is_deepseek_models(self):
        """TOGETHER_MODELS should be the same list as TOGETHER_DEEPSEEK_MODELS."""
        assert TOGETHER_MODELS is TOGETHER_DEEPSEEK_MODELS

    def test_model_icons(self):
        for model in TOGETHER_DEEPSEEK_MODELS:
            assert model.model_icon is not None
            assert "deepseek" in model.model_icon

    def test_tokenizer(self):
        for model in TOGETHER_DEEPSEEK_MODELS:
            assert model.tokenizer_name == "cl100k_base"

    def test_highlights(self):
        for model in TOGETHER_DEEPSEEK_MODELS:
            assert len(model.highlights) > 0
