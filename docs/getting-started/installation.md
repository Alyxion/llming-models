# Installation

## From PyPI

```bash
pip install llming-models
```

### Optional Extras

For MongoDB-backed budget persistence:

```bash
pip install llming-models[mongodb]
```

## From Source (Poetry)

```bash
git clone https://github.com/Alyxion/llming-models.git
cd llming-models
poetry install
```

To include MongoDB support from source:

```bash
poetry install -E mongodb
```

## Environment Setup

llming-models discovers providers via environment variables. Set the keys for the providers you want to use:

```bash
# Create .env from template
cp .env.template .env
```

### Required Keys (at least one)

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI (GPT-5.x models) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude models) |
| `GOOGLE_API_KEY` | Google (Gemini models) |
| `MISTRAL_API_KEY` | Mistral |
| `TOGETHER_API_KEY` | Together AI / DeepSeek |

### Azure Keys (optional)

| Variable | Provider |
|---|---|
| `AZURE_OPENAI_API_KEY` | Azure-hosted OpenAI |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_ANTHROPIC_API_KEY` | Azure-hosted Anthropic |

### Media Keys (optional)

| Variable | Service |
|---|---|
| `OPENAI_API_KEY` | OpenAI TTS/STT (shared with LLM key) |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS/STT |

!!! tip "Loading .env files"
    Use `python-dotenv` to load `.env` files automatically:

    ```python
    from dotenv import load_dotenv
    load_dotenv()
    ```

## Dependencies

Core dependencies are installed automatically:

- `pydantic` -- data models and validation
- `tiktoken` -- token counting
- `openai` >= 2.14 -- OpenAI client
- `anthropic` -- Anthropic client
- `google-genai` >= 1.56 -- Google Gemini client
- `httpx` >= 0.27 -- HTTP client (ElevenLabs)
- `mcp` >= 1.0 -- Model Context Protocol

## Verify Installation

```python
import llming_models
print(llming_models.__all__)
```

This should print the list of exported symbols without errors.
