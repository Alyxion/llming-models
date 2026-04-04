"""Azure Anthropic model configurations.

Deployment names map to Azure AI Services deployment names.
These mirror the standard Anthropic models but are hosted in Azure.
"""
from __future__ import annotations

import os
from typing import Any

from ..llm_provider_models import LLMInfo, ModelSize


def get_azure_anthropic_models() -> list[LLMInfo]:
    """Build model list from environment.

    Reads AZURE_ANTHROPIC_DEPLOYMENTS to determine which models are available.
    Format: comma-separated name=deployment pairs.

    Example:
        AZURE_ANTHROPIC_DEPLOYMENTS=claude_opus=claude-opus-4-6-2,claude_sonnet=claude-opus-4-6-2

    This routes both Opus and Sonnet requests through the claude-opus-4-6-2
    deployment. Useful when only one deployment has quota.
    """
    raw = os.environ.get("AZURE_ANTHROPIC_DEPLOYMENTS", "").strip()
    if not raw:
        return []

    # Base definitions keyed by canonical name
    model_defs: dict[str, dict[str, Any]] = {
        "claude_opus": dict(
            label="Claude Opus 4.6",
            description="Most capable Claude model for complex tasks.",
            input_token_price=5.00,
            cached_input_token_price=0.50,
            output_token_price=25.00,
            size=ModelSize.LARGE,
            max_input_tokens=200000,
            max_output_tokens=128000,
            popularity=101,
            speed=5,
            quality=10,
            best_use="Deep analysis",
            highlights=["Best reasoning", "Code", "Analysis", "Web search"],
        ),
        "claude_sonnet": dict(
            label="Claude Sonnet 4.6",
            description="Flagship-level performance at mid-tier pricing.",
            input_token_price=3.00,
            cached_input_token_price=0.30,
            output_token_price=15.00,
            size=ModelSize.MEDIUM,
            max_input_tokens=200000,
            max_output_tokens=64000,
            popularity=88,
            speed=7,
            quality=9,
            best_use="Analysis & code",
            highlights=["Reasoning", "Code", "Analysis", "Web search"],
        ),
        "claude_haiku": dict(
            label="Claude Haiku 4.5",
            description="Fastest model with near-frontier intelligence.",
            input_token_price=1.00,
            cached_input_token_price=0.10,
            output_token_price=5.00,
            size=ModelSize.SMALL,
            max_input_tokens=200000,
            max_output_tokens=64000,
            popularity=75,
            speed=10,
            quality=7,
            best_use="Quick tasks",
            highlights=["Fast", "Code", "Web search"],
        ),
    }

    models = []
    for entry in raw.split(","):
        entry = entry.strip()
        if "=" not in entry:
            continue
        name, deployment = entry.split("=", 1)
        name = name.strip()
        deployment = deployment.strip()
        if name not in model_defs:
            continue
        d = model_defs[name]
        models.append(LLMInfo(
            provider="azure_anthropic",
            name=name,
            label=d["label"],
            model=deployment,
            description=d["description"],
            input_token_price=d["input_token_price"],
            cached_input_token_price=d["cached_input_token_price"],
            output_token_price=d["output_token_price"],
            model_icon="models/claude-240.svg",
            company_icon="companies/Anthropic_logo.svg",
            hosting_icon="companies/azure.svg",
            size=d["size"],
            max_input_tokens=d["max_input_tokens"],
            max_output_tokens=d["max_output_tokens"],
            popularity=d["popularity"],
            speed=d["speed"],
            quality=d["quality"],
            best_use=d["best_use"],
            highlights=d["highlights"],
            supports_image_input=True,
            reasoning=True,
            default_tools=["web_search"],
        ))

    return models


AZURE_ANTHROPIC_MODELS = get_azure_anthropic_models()
