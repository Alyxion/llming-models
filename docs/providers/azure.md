# Azure Provider

llming-models supports Azure-hosted versions of OpenAI and Anthropic models for enterprise deployments.

## Azure OpenAI

### Configuration

```python
from llming_models import ChatSession, LLMConfig

session = ChatSession(
    config=LLMConfig(
        provider="azure_openai",
        model="gpt-5.2",  # Azure deployment name
        base_url="https://your-resource.openai.azure.com/",
    ),
)
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI resource key |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint URL |

## Azure Anthropic

### Configuration

```python
session = ChatSession(
    config=LLMConfig(
        provider="azure_anthropic",
        model="claude-sonnet-4-6",
    ),
)
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `AZURE_ANTHROPIC_API_KEY` | Yes | Azure Anthropic resource key |

## Tool Compatibility

Azure providers are API-compatible with their base providers. Tools and configurations that work with `openai` automatically work with `azure_openai`, and `anthropic` tools work with `azure_anthropic`:

```python
from llming_models.tools.tool_definition import PROVIDER_COMPAT

# Automatic mapping:
# "azure_openai" -> "openai"
# "azure_anthropic" -> "anthropic"
```

This means web search, image generation, and MCP tools work identically across standard and Azure providers.

!!! note "Enterprise features"
    Azure providers are ideal for organizations requiring VNet integration, private endpoints, managed identity, and compliance certifications. The model capabilities are identical to their non-Azure counterparts.
