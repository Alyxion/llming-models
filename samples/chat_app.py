"""Interactive chat playground for llming-models.

A FastAPI app with vanilla JS frontend for testing LLM providers and models.
Run:  cd /path/to/llming-models && set -a && source .env && set +a && python samples/chat_app.py
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Ensure the package is importable when running from the samples directory
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from llming_models import LLMManager, ChatSession
from llming_models.config import LLMUserConfig
from llming_models.llm_base_models import Role
from llming_models.media import MediaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="llming playground")

# Serve static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global manager — initialised on startup
manager: LLMManager | None = None
_media: MediaManager | None = None


@app.on_event("startup")
async def startup():
    global manager, _media
    manager = LLMManager(user_config=LLMUserConfig())
    _media = MediaManager()
    providers = list(manager.providers.keys())
    media_providers = [p.provider_name for p in _media._tts_providers]
    logger.info("LLMManager ready  providers=%s", providers)
    logger.info("MediaManager ready  media_providers=%s", media_providers)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat UI."""
    html_path = STATIC_DIR / "chat.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/models")
async def list_models():
    """Return available models grouped by provider."""
    assert manager is not None
    models = manager.get_available_llms()
    grouped: dict[str, list[dict]] = {}
    for m in models:
        entry = {
            "name": m.name,
            "label": m.label,
            "model": m.model,
            "provider": m.provider,
            "description": m.description,
            "input_token_price": m.input_token_price,
            "output_token_price": m.output_token_price,
            "max_input_tokens": m.max_input_tokens,
            "max_output_tokens": m.max_output_tokens,
            "speed": m.speed,
            "quality": m.quality,
            "best_use": m.best_use,
            "highlights": m.highlights,
            "reasoning": m.reasoning,
            "supports_image_input": m.supports_image_input,
        }
        grouped.setdefault(m.provider, []).append(entry)
    return JSONResponse(content=grouped)


def _build_session(provider: str, model: str, temperature: float = 0.7,
                   system_prompt: str | None = None) -> ChatSession:
    """Create a ChatSession for the given provider:model combo."""
    assert manager is not None
    config = manager.get_config_for_model(f"{provider}:{model}")
    config.temperature = temperature
    creds = manager.credentials.for_provider(provider) if manager.credentials else None
    return ChatSession(config=config, system_prompt=system_prompt, credentials=creds)


@app.post("/api/chat")
async def chat(request: Request):
    """Non-streaming chat endpoint.

    Body: {provider, model, messages: [{role, content}], temperature?, system_prompt?}
    """
    body = await request.json()
    provider = body["provider"]
    model_name = body["model"]
    messages: list[dict] = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    system_prompt = body.get("system_prompt")

    session = _build_session(provider, model_name, temperature, system_prompt)

    # Replay history (all messages except the last one)
    for msg in messages[:-1]:
        session.add_message(Role(msg["role"]), msg["content"])

    # Send the last user message
    last_content = messages[-1]["content"] if messages else ""
    try:
        response = await session.chat_async(last_content, streaming=False)
        metadata = response.response_metadata or {}
        return JSONResponse(content={
            "text": response.content,
            "usage": {
                "input_tokens": metadata.get("input_tokens") or metadata.get("total_input_tokens", 0),
                "output_tokens": metadata.get("output_tokens") or metadata.get("total_output_tokens", 0),
                "cached_input_tokens": metadata.get("cached_input_tokens", 0),
            },
        })
    except Exception as exc:
        logger.exception("Chat error")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """Streaming chat endpoint — returns Server-Sent Events."""
    body = await request.json()
    provider = body["provider"]
    model_name = body["model"]
    messages: list[dict] = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    system_prompt = body.get("system_prompt")

    session = _build_session(provider, model_name, temperature, system_prompt)

    # Replay history (all except the last message)
    for msg in messages[:-1]:
        session.add_message(Role(msg["role"]), msg["content"])

    last_content = messages[-1]["content"] if messages else ""

    async def event_generator():
        try:
            stream = await session.chat_async(last_content, streaming=True)
            async for chunk in stream:
                if chunk.content:
                    payload = json.dumps({"text": chunk.content})
                    yield f"data: {payload}\n\n"
                # On the final chunk, include usage metadata
                if chunk.is_final:
                    metadata = chunk.response_metadata or {}
                    usage = {
                        "input_tokens": metadata.get("input_tokens") or metadata.get("total_input_tokens", 0),
                        "output_tokens": metadata.get("output_tokens") or metadata.get("total_output_tokens", 0),
                        "cached_input_tokens": metadata.get("cached_input_tokens", 0),
                    }
                    yield f"data: {json.dumps({'done': True, 'usage': usage})}\n\n"
        except Exception as exc:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# TTS / STT Routes
# ---------------------------------------------------------------------------

@app.get("/api/voices")
async def list_voices():
    """Return available TTS voices from all providers."""
    assert _media is not None
    return [v.model_dump() for v in _media.list_all_voices()]


@app.get("/api/tts/providers")
async def tts_providers():
    """Return available media provider names and model info."""
    assert _media is not None
    providers = [p.provider_name for p in _media._tts_providers]
    default_provider = _media.tts.provider_name if _media._tts_providers else None
    models = _media.get_all_models()
    return {
        "providers": providers,
        "default": default_provider,
        "models": [
            {
                "provider": m.provider,
                "name": m.name,
                "label": m.label,
                "media_type": m.media_type.value,
                "price_per_1m_chars": m.price_per_1m_chars,
                "price_per_minute": m.price_per_minute,
                "supports_word_timings": m.supports_word_timings,
                "speed": m.speed,
                "quality": m.quality,
            }
            for m in models
        ],
    }


@app.post("/api/tts")
async def text_to_speech(request: Request):
    """Synthesize speech from text. Returns base64 audio + word timings + cost."""
    assert _media is not None
    body = await request.json()
    text = body.get("text", "")
    voice = body.get("voice", "")
    provider_name = body.get("provider", "")
    with_timings = body.get("with_timings", False)

    try:
        provider = _media.get_provider(provider_name)
        estimated_cost = provider.estimate_tts_cost(text)
        result = await provider.synthesize(
            text, voice=voice, with_timings=with_timings,
        )
        return {
            "audio_b64": base64.b64encode(result.audio_bytes).decode(),
            "content_type": result.content_type,
            "word_timings": [t.model_dump() for t in result.word_timings],
            "duration": result.duration,
            "estimated_cost": estimated_cost,
        }
    except Exception as exc:
        logger.exception("TTS error")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...), provider: str = ""):
    """Transcribe an uploaded audio file to text."""
    assert _media is not None
    try:
        audio_bytes = await file.read()
        p = _media.get_provider(provider) if provider else _media.stt
        result = await p.transcribe(
            audio_bytes,
            filename=file.filename or "audio.webm",
            content_type=file.content_type or "audio/webm",
            with_timings=True,
        )
        return {
            "text": result.text,
            "word_timings": [t.model_dump() for t in result.word_timings],
        }
    except Exception as exc:
        logger.exception("STT error")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("chat_app:app", host="0.0.0.0", port=port, reload=True,
                app_dir=str(Path(__file__).resolve().parent))
