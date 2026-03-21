"""Tests for llming_models.model_info — LLMInfo, ModelSize, ReasoningEffort."""

from llming_models.model_info import LLMInfo, ModelSize, ReasoningEffort


class TestModelSize:
    def test_ordering(self):
        assert ModelSize.VERY_SMALL < ModelSize.SMALL < ModelSize.MEDIUM < ModelSize.LARGE < ModelSize.VERY_LARGE

    def test_values(self):
        assert ModelSize.VERY_SMALL == 1
        assert ModelSize.VERY_LARGE == 5


class TestReasoningEffort:
    def test_values(self):
        assert ReasoningEffort.NONE.value == "none"
        assert ReasoningEffort.HIGH.value == "high"

    def test_is_string(self):
        assert isinstance(ReasoningEffort.MEDIUM, str)
        assert ReasoningEffort.MEDIUM == "medium"


class TestLLMInfo:
    def test_basic_creation(self):
        info = LLMInfo(
            provider="test",
            name="test_model",
            label="Test Model",
            model="test-v1",
            description="A test model",
            input_token_price=1.0,
        )
        assert info.provider == "test"
        assert info.model == "test-v1"
        assert info.size == ModelSize.MEDIUM  # default
        assert info.max_input_tokens == 64000  # default
        assert info.supports_system_prompt is True  # default

    def test_full_creation(self):
        info = LLMInfo(
            provider="anthropic",
            name="claude_sonnet",
            label="Claude Sonnet",
            model="claude-sonnet-4-6",
            description="Fast model",
            input_token_price=3.0,
            cached_input_token_price=0.3,
            output_token_price=15.0,
            size=ModelSize.MEDIUM,
            max_input_tokens=200_000,
            max_output_tokens=64_000,
            supports_image_input=True,
            reasoning=True,
            default_reasoning_effort=ReasoningEffort.MEDIUM,
            speed=8,
            quality=8,
            best_use="Code",
            highlights=["Fast", "Code"],
            default_tools=["web_search"],
        )
        assert info.output_token_price == 15.0
        assert info.cached_input_token_price == 0.3
        assert info.supports_image_input is True
        assert info.reasoning is True
        assert info.default_reasoning_effort == ReasoningEffort.MEDIUM
        assert info.highlights == ["Fast", "Code"]
        assert info.default_tools == ["web_search"]

    def test_defaults(self):
        info = LLMInfo(
            provider="p", name="n", label="l", model="m",
            description="d", input_token_price=0.0,
        )
        assert info.cached_input_token_price == 0.0
        assert info.output_token_price == 0.0
        assert info.api_base is None
        assert info.tokenizer_name is None
        assert info.model_icon is None
        assert info.popularity == 0
        assert info.reasoning is False
        assert info.reasoning_effort is None
        assert info.enforced_temperature is None
        assert info.supports_image_input is False
        assert info.highlights == []
        assert info.default_tools == []
        assert info.native_tools == {}
