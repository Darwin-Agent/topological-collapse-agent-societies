"""
Centralized model configuration for all LLM experiments.
16 models from 9 vendors. Newest available versions only.
Updated: 2026-04-27
"""

import os
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str           # Short display name
    model_id: str       # API model ID
    vendor: str         # Company name
    release: str        # Approximate release date


# ── Final 16 models: newest per vendor, no redundancy ─────────────

MODELS_16: list[ModelSpec] = [
    # --- OpenAI (2: latest flagship + reasoning) ---
    ModelSpec("GPT-5.4",        "gpt-5.4",        "OpenAI",    "2026-03"),
    ModelSpec("o4-mini",        "o4-mini",        "OpenAI",    "2026-02"),
    # --- Anthropic (2: opus + sonnet) ---
    ModelSpec("Claude-Opus-4.5",   "claude-opus-4-5",   "Anthropic", "2025-11"),
    ModelSpec("Claude-Sonnet-4.5", "claude-sonnet-4-5", "Anthropic", "2025-09"),
    # --- DeepSeek (2: flagship + reasoning) ---
    ModelSpec("DeepSeek-V3.2",  "deepseek-v3.2", "DeepSeek", "2025-12"),
    ModelSpec("DeepSeek-R1",    "deepseek-r1",   "DeepSeek", "2025-11"),
    # --- Alibaba Qwen (2: latest) ---
    ModelSpec("Qwen3.6-plus",   "qwen3.6-plus", "Alibaba", "2026-04"),
    ModelSpec("Qwen3.5-plus",   "qwen3.5-plus", "Alibaba", "2026-02"),
    # --- Zhipu (1) ---
    ModelSpec("GLM-5",          "glm-5", "Zhipu", "2025-12"),
    # --- Moonshot (2) ---
    ModelSpec("Kimi-K2.5",      "kimi-k2.5",          "Moonshot", "2026-01"),
    ModelSpec("Kimi-K2",        "kimi-k2-instruct",   "Moonshot", "2025-11"),
    # --- MiniMax (1) ---
    ModelSpec("MiniMax-M2.7",   "minimax-m2.7", "MiniMax", "2026-03"),
    # --- Xiaomi (1) ---
    ModelSpec("MiMo-v2.5-Pro",  "mimo-v2.5-pro", "Xiaomi", "2026-02"),
    # --- ByteDance (2) ---
    ModelSpec("Seed-OSS-36B",   "seed-oss-36b-instruct", "ByteDance", "2025-12"),
    # --- Extra capacity ---
    ModelSpec("Qwen3-max",      "qwen3-max", "Alibaba", "2025-12"),
    ModelSpec("GPT-5-mini",     "gpt-5-mini", "OpenAI", "2025-09"),
]

MODELS_BY_NAME: dict[str, ModelSpec] = {m.name: m for m in MODELS_16}

API_URL_ENV = "LLM_GATEWAY_API_URL"
API_KEY_ENV = "LLM_GATEWAY_API_KEY"


def get_api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} must be set in the environment.")
    return api_key


def get_api_url() -> str:
    api_url = os.environ.get(API_URL_ENV, "")
    if not api_url:
        raise RuntimeError(
            f"{API_URL_ENV} must be set to an OpenAI-compatible "
            "chat-completions endpoint."
        )
    return api_url


def get_headers(spec: ModelSpec) -> dict:
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }


if __name__ == "__main__":
    print(f"Total models: {len(MODELS_16)}")
    vendors = set(m.vendor for m in MODELS_16)
    print(f"Vendors ({len(vendors)}): {', '.join(sorted(vendors))}")
    for i, m in enumerate(MODELS_16):
        print(f"  [{i:2d}] {m.name:20s} | {m.vendor:10s} | {m.release} | {m.model_id}")
