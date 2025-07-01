from __future__ import annotations

import os
import logging
import asyncio
from typing import AsyncIterator, Dict, Any

"""Light abstraction layer over one or more LLM back-ends.

Right now we only implement Google Gemini via `google-genai`, but the
interface allows us to add more providers later without touching UI code.
"""

# Configure logger for LLM interactions (disabled by default)
logger = logging.getLogger("spegel.llm")
logger.setLevel(logging.CRITICAL + 1)  # Effectively disabled by default

def enable_llm_logging(level: int = logging.INFO) -> None:
    """Enable LLM interaction logging at the specified level."""
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover – dependency is optional until used
    genai = None  # type: ignore

try:
    import anthropic
except ImportError:  # pragma: no cover – dependency is optional until used
    anthropic = None  # type: ignore

__all__ = [
    "LLMClient",
    "GeminiClient",
    "ClaudeClient",
    "get_default_client",
]


class LLMClient:
    """Abstract asynchronous client interface."""

    async def stream(self, prompt: str, content: str, **kwargs) -> AsyncIterator[str]:
        """Yield chunks of markdown text."""
        raise NotImplementedError
        yield # This is unreachable, but makes this an async generator


class GeminiClient(LLMClient):
    """Wrapper around google-genai async streaming API."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        if genai is None:
            raise RuntimeError("google-genai not installed but GeminiClient requested")
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def stream(
        self,
        prompt: str,
        content: str,
        generation_config: Dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if generation_config is None:
            generation_config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens =8192,
                response_mime_type="text/plain",
                thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        )
            )
        user_content = f"{prompt}\n\n{content}" if content else prompt
        stream = self._client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=user_content,
            config=generation_config,
        )

        # Log the prompt if logging is enabled
        logger.info("LLM Prompt: %s", user_content)

        collected: list[str] = []

        async for chunk in await stream:
            try:
                text = chunk.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
                if text:
                    collected.append(text)
                    yield text
            except Exception:
                continue

        # Log the complete response if logging is enabled
        if collected:
            logger.info("LLM Response: %s", "".join(collected))


class ClaudeClient(LLMClient):
    """Wrapper around Anthropic Claude async streaming API."""

    def __init__(self, api_key: str, model_name: str = "claude-3-5-haiku-20241022"):
        if anthropic is None:
            raise RuntimeError("anthropic not installed but ClaudeClient requested")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model_name = model_name

    async def stream(
        self,
        prompt: str,
        content: str,
        max_retries: int = 3,
        **kwargs
    ) -> AsyncIterator[str]:
        user_content = f"{prompt}\n\n{content}" if content else prompt
        
        # Log the prompt if logging is enabled
        logger.info("LLM Prompt: %s", user_content)

        collected: list[str] = []

        for attempt in range(max_retries + 1):
            try:
                async with self._client.messages.stream(
                    model=self.model_name,
                    max_tokens=8192,
                    temperature=0.2,
                    messages=[{"role": "user", "content": user_content}]
                ) as stream:
                    async for text in stream.text_stream:
                        if text:
                            collected.append(text)
                            yield text
                
                # If we get here, streaming succeeded
                break
                
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(keyword in error_str for keyword in [
                    'overloaded', 'rate limit', 'too many requests', 
                    'service unavailable', 'timeout', 'temporary'
                ])
                
                if is_retryable and attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = 2 ** attempt
                    logger.warning(
                        "Claude API error (attempt %d/%d): %s. Retrying in %ds...", 
                        attempt + 1, max_retries + 1, e, delay
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable error or max retries exceeded
                    logger.error("Claude streaming error (final): %s", e)
                    yield f"Error: {e}"
                    break

        # Log the complete response if logging is enabled
        if collected:
            logger.info("LLM Response: %s", "".join(collected))


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_default_client(provider: str = "auto") -> tuple[LLMClient | None, bool]:
    """Return an LLMClient instance if credentials exist, else (None, False).
    
    Args:
        provider: "auto", "claude", or "gemini". Auto checks Claude first, then Gemini.
    """
    if provider == "claude":
        # Force Claude only
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic is not None:
            return ClaudeClient(anthropic_key), True
        return None, False
    
    elif provider == "gemini":
        # Force Gemini only
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and genai is not None:
            return GeminiClient(gemini_key), True
        return None, False
    
    else:  # provider == "auto"
        # Check for Claude first
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic is not None:
            return ClaudeClient(anthropic_key), True
        
        # Fall back to Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and genai is not None:
            return GeminiClient(gemini_key), True
            
    return None, False


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        description="Quick CLI wrapper around the configured LLM to answer a prompt."
    )
    parser.add_argument("prompt", help="User prompt/question to send to the model")
    args = parser.parse_args()

    client, ok = get_default_client()
    if not ok or client is None:
        print("Error: GEMINI_API_KEY not set or google-genai unavailable", file=sys.stderr)
        sys.exit(1)

    async def _main() -> None:
        async for chunk in client.stream(args.prompt, ""):
            print(chunk, end="", flush=True)
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass 