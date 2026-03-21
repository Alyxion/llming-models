"""Model catalog example.

Shows how to define and query model metadata.
"""

from llming_models import LLMInfo, ModelSize, ReasoningEffort

MODELS = [
    LLMInfo(
        provider="anthropic",
        name="claude_sonnet",
        label="Claude Sonnet 4.6",
        model="claude-sonnet-4-6",
        description="Fast, capable model for most tasks",
        input_token_price=3.0,
        output_token_price=15.0,
        size=ModelSize.MEDIUM,
        max_input_tokens=200_000,
        max_output_tokens=64_000,
        supports_image_input=True,
        reasoning=True,
        speed=8,
        quality=8,
        highlights=["Fast", "Code", "Vision"],
    ),
    LLMInfo(
        provider="anthropic",
        name="claude_haiku",
        label="Claude Haiku 4.5",
        model="claude-haiku-4-5-20251001",
        description="Fastest, most affordable model",
        input_token_price=0.80,
        output_token_price=4.0,
        size=ModelSize.SMALL,
        max_input_tokens=200_000,
        max_output_tokens=8_192,
        supports_image_input=True,
        speed=10,
        quality=6,
        highlights=["Fastest", "Affordable"],
    ),
    LLMInfo(
        provider="openai",
        name="gpt_4o",
        label="GPT-4o",
        model="gpt-4o",
        description="Most capable OpenAI model",
        input_token_price=2.50,
        output_token_price=10.0,
        size=ModelSize.LARGE,
        max_input_tokens=128_000,
        max_output_tokens=16_384,
        supports_image_input=True,
        speed=7,
        quality=9,
        highlights=["Versatile", "Vision", "Web search"],
    ),
]


def main():
    print("=== Model Catalog ===\n")

    for m in MODELS:
        cost_1k = (1000 * m.input_token_price + 1000 * m.output_token_price) / 1_000_000
        print(f"{m.label} ({m.provider})")
        print(f"  Model ID:  {m.model}")
        print(f"  Size:      {m.size.name}")
        print(f"  Context:   {m.max_input_tokens:,} in / {m.max_output_tokens:,} out")
        print(f"  Pricing:   ${m.input_token_price}/1M in, ${m.output_token_price}/1M out")
        print(f"  ~Cost/1K:  ${cost_1k:.4f}")
        print(f"  Speed:     {m.speed}/10  Quality: {m.quality}/10")
        print(f"  Vision:    {m.supports_image_input}  Reasoning: {m.reasoning}")
        print(f"  Highlights: {', '.join(m.highlights)}")
        print()

    # Filter models
    vision_models = [m for m in MODELS if m.supports_image_input]
    print(f"Vision-capable models: {[m.label for m in vision_models]}")

    cheapest = min(MODELS, key=lambda m: m.input_token_price)
    print(f"Cheapest model: {cheapest.label} (${cheapest.input_token_price}/1M)")


if __name__ == "__main__":
    main()
