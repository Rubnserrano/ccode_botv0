"""LLM Client — OpenRouter API wrapper with fallback chain.

Minimizes costs: starts with free model (deepseek-chat),
falls back to paid models only if needed.

Usage:
    from src.brain.llm_client import LLMClient
    client = LLMClient(api_key="sk-or-...")
    response = await client.generate("system prompt", "user prompt")
    await client.close()
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cost chain: cheapest first
MODEL_CHAIN = [
    {"name": "deepseek/deepseek-chat",       "cost_per_1k_in": 0.0,   "cost_per_1k_out": 0.0},
    {"name": "anthropic/claude-3-haiku",     "cost_per_1k_in": 0.00025, "cost_per_1k_out": 0.00125},
    {"name": "openai/gpt-4o-mini",           "cost_per_1k_in": 0.00015, "cost_per_1k_out": 0.0006},
]

AUDIT_DIR = Path(__file__).resolve().parents[2] / "data" / "parquet" / "brain"


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + user).encode()).hexdigest()[:16]


def _audit_path() -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIT_DIR / "audit.jsonl"


def _append_audit(entry: dict) -> None:
    path = _audit_path()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


class LLMClient:
    """Async LLM client with model fallback, cost tracking, and audit log.

    Each call is logged to data/parquet/brain/audit.jsonl for future training.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ""
        self.client = httpx.AsyncClient(timeout=60)
        self._cache: dict[str, dict] = {}
        self.total_cost = 0.0

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = "json_object",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """Call OpenRouter with fallback chain.

        Returns parsed JSON response dict with keys:
          - "content": the text/JSON response
          - "model": model used
          - "cost": estimated cost
          - "cached": True if from cache
        """
        # In-memory cache
        cache_key = _prompt_hash(system_prompt, user_prompt)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return {**cached, "cached": True}

        last_error = None
        for model_info in MODEL_CHAIN:
            model = model_info["name"]
            max_retries = 4 if model == MODEL_CHAIN[0]["name"] else 0
            for attempt in range(max_retries + 1):
                try:
                    result = await self._call_model(
                        model, system_prompt, user_prompt,
                        response_format, max_tokens, temperature,
                    )
                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code == 429 and attempt < max_retries:
                        wait = 5 * (2 ** attempt)
                        logger.warning("brain: %s rate limited, retry in %ds", model, wait)
                        await asyncio.sleep(wait)
                        continue
                    logger.warning("brain: %s failed (%s)", model, e)
                    break  # try next model

                # Success — cost, extract, cache
                in_tokens = result.get("usage", {}).get("prompt_tokens", 0)
                out_tokens = result.get("usage", {}).get("completion_tokens", 0)
                cost = (
                    in_tokens / 1000 * model_info["cost_per_1k_in"]
                    + out_tokens / 1000 * model_info["cost_per_1k_out"]
                )
                self.total_cost += cost
                content_str = result["choices"][0]["message"]["content"]

                if response_format == "json_object":
                    if "```json" in content_str:
                        content_str = content_str.split("```json")[1].split("```")[0]
                    elif "```" in content_str:
                        content_str = content_str.split("```")[1].split("```")[0]

                response_data = {
                    "content": content_str, "model": model,
                    "cost": round(cost, 6), "cached": False,
                    "tokens_in": in_tokens, "tokens_out": out_tokens,
                }
                self._cache[cache_key] = response_data
                _append_audit({
                    "ts": time.time(), "prompt_hash": cache_key,
                    "model": model, "system_prompt": system_prompt,
                    "user_prompt": user_prompt, "response": content_str,
                    "tokens_in": in_tokens, "tokens_out": out_tokens,
                    "cost": cost, "cached": False,
                })
                logger.info("brain: %s  in=%d out=%d cost=$%.6f",
                            model, in_tokens, out_tokens, cost)
                return response_data

            else:
                # Inner for loop completed without success (rate limit exhausted)
                continue

        raise RuntimeError(f"All LLM models failed. Last error: {last_error}")

    async def _call_model(
        self, model: str, system: str, user: str,
        response_format: str | None, max_tokens: int, temperature: float,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages or [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = {"type": response_format}
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = await self.client.post(OPENROUTER_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self.client.aclose()
