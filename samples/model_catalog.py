"""Model catalog — discover and compare all available LLM models.

Automatically detects which providers are configured (via API keys in
environment variables or .env) and lists every model with pricing,
capabilities, and context window sizes.

    cp .env.template .env   # fill in your API keys
    python samples/model_catalog.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from llming_models import LLMManager, LLMUserConfig


def main() -> None:
    manager = LLMManager(user_config=LLMUserConfig())

    if not manager.providers:
        print("No providers available. Set API keys in .env (see .env.template).")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_KEY, MISTRAL_API_KEY, TOGETHER_API_KEY")
        sys.exit(1)

    all_models = []
    for name, provider in manager.providers.items():
        models = provider.get_models()
        all_models.extend((name, m) for m in models)

    print(f"=== {len(all_models)} Models from {len(manager.providers)} Providers ===\n")
    print(f"{'Model':<35} {'Provider':<12} {'Size':<8} {'Context':>12}  {'$/1M in':>8} {'$/1M out':>9}  {'Spd':>3} {'Qlty':>4}  {'Capabilities'}")
    print("-" * 130)

    for provider_name, m in sorted(all_models, key=lambda x: (x[0], x[1].input_token_price)):
        ctx = f"{m.max_input_tokens // 1000}K"
        caps = []
        if m.supports_image_input:
            caps.append("vision")
        if m.reasoning:
            caps.append("reasoning")
        if m.highlights:
            caps.extend(h.lower() for h in m.highlights[:2])
        cap_str = ", ".join(caps) if caps else ""

        print(
            f"{m.label:<35} {provider_name:<12} {m.size.name:<8} {ctx:>12}  "
            f"${m.input_token_price:>7.2f} ${m.output_token_price:>8.2f}  "
            f"{m.speed:>3} {m.quality:>4}  {cap_str}"
        )

    # Summary
    print(f"\n{'='*130}")
    print(f"Providers: {', '.join(manager.providers.keys())}")

    cheapest = min(all_models, key=lambda x: x[1].input_token_price)
    fastest = max(all_models, key=lambda x: x[1].speed)
    best_quality = max(all_models, key=lambda x: x[1].quality)
    biggest_ctx = max(all_models, key=lambda x: x[1].max_input_tokens)

    print(f"Cheapest:  {cheapest[1].label} (${cheapest[1].input_token_price}/1M)")
    print(f"Fastest:   {fastest[1].label} (speed {fastest[1].speed}/10)")
    print(f"Best:      {best_quality[1].label} (quality {best_quality[1].quality}/10)")
    print(f"Largest:   {biggest_ctx[1].label} ({biggest_ctx[1].max_input_tokens // 1000}K context)")

    vision = [m.label for _, m in all_models if m.supports_image_input]
    reasoning = [m.label for _, m in all_models if m.reasoning]
    print(f"Vision:    {len(vision)} models — {', '.join(vision[:5])}")
    print(f"Reasoning: {len(reasoning)} models — {', '.join(reasoning[:5])}")


if __name__ == "__main__":
    main()
