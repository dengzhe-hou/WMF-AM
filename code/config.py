"""
Shared configuration for all CEF experiment protocols.

Set API keys in a .env file at the project root:
    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...
    TOGETHER_API_KEY=...
    OPENROUTER_API_KEY=...
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Model registry ──────────────────────────────────────────────────────────
# 12 models spanning 5 provider families (expanded from 5 for E5 statistical validity).
# N=12 enables Kendall's tau rank correlation (p-values reliable at N>=10).

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

MODELS = {
    # Ollama (local, no API key needed)
    "ollama:qwen2.5:7b": {
        "provider": "ollama",
        "model_id": "qwen2.5:7b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:qwen2.5:14b": {
        "provider": "ollama",
        "model_id": "qwen2.5:14b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:qwen2.5:32b": {
        "provider": "ollama",
        "model_id": "qwen2.5:32b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:llama3.1:70b": {
        "provider": "ollama",
        "model_id": "llama3.1:70b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:gemma2:27b": {
        "provider": "ollama",
        "model_id": "gemma2:27b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:deepseek-r1:14b": {
        "provider": "ollama",
        "model_id": "deepseek-r1:14b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:mistral:7b": {
        "provider": "ollama",
        "model_id": "mistral:7b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:llama3.1:8b": {
        "provider": "ollama",
        "model_id": "llama3.1:8b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # ── Additional Ollama models for N-expansion ──
    "ollama:phi3:14b": {
        "provider": "ollama",
        "model_id": "phi3:14b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:gemma2:9b": {
        "provider": "ollama",
        "model_id": "gemma2:9b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:qwen2.5:3b": {
        "provider": "ollama",
        "model_id": "qwen2.5:3b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:llama3.2:3b": {
        "provider": "ollama",
        "model_id": "llama3.2:3b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:deepseek-r1:7b": {
        "provider": "ollama",
        "model_id": "deepseek-r1:7b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:mixtral:8x7b": {
        "provider": "ollama",
        "model_id": "mixtral:8x7b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:command-r:35b": {
        "provider": "ollama",
        "model_id": "command-r:35b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:yi:34b": {
        "provider": "ollama",
        "model_id": "yi:34b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # ── Very small models for broader baseline (range extension) ──
    "ollama:qwen2.5:0.5b": {
        "provider": "ollama",
        "model_id": "qwen2.5:0.5b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:qwen2.5:1.5b": {
        "provider": "ollama",
        "model_id": "qwen2.5:1.5b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:llama3.2:1b": {
        "provider": "ollama",
        "model_id": "llama3.2:1b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:gemma2:2b": {
        "provider": "ollama",
        "model_id": "gemma2:2b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "ollama:tinyllama:1.1b": {
        "provider": "ollama",
        "model_id": "tinyllama:1.1b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # OpenAI
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Anthropic
    "claude-opus-4": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Google
    "gemini-1.5-pro": {
        "provider": "google",
        "model_id": "gemini-1.5-pro",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "model_id": "gemini-1.5-flash",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Together.ai (Meta)
    "llama3-70b": {
        "provider": "together",
        "model_id": "meta-llama/Llama-3-70b-chat-hf",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "llama3-8b": {
        "provider": "together",
        "model_id": "meta-llama/Llama-3-8b-chat-hf",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Together.ai (Mistral)
    "mixtral-8x7b": {
        "provider": "together",
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Together.ai (Qwen)
    "qwen2-72b": {
        "provider": "together",
        "model_id": "Qwen/Qwen2-72B-Instruct",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Together.ai (DeepSeek)
    "deepseek-v3": {
        "provider": "together",
        "model_id": "deepseek-ai/DeepSeek-V3",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Google (Gemma)
    "gemma-2-27b": {
        "provider": "google",
        "model_id": "gemma-2-27b-it",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # ── Held-out API models (expansion, N=28) ───────────────────────────────
    # Standard models via OpenRouter
    "openrouter:gpt-4o": {
        "provider": "openrouter",
        "model_id": "openai/gpt-4o",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:gpt-4o-mini": {
        "provider": "openrouter",
        "model_id": "openai/gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:claude-sonnet-4": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:deepseek-v3": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-chat-v3-0324",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # LRM (Long Reasoning Models) via OpenRouter
    "openrouter:o3-mini": {
        "provider": "openrouter",
        "model_id": "openai/o3-mini",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:deepseek-r1": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Google via free AI Studio API
    # Google free tier has 5 RPM limit — use OpenRouter instead ($0.30/M input)
    "openrouter:gemini-2.5-flash": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-flash",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:gemini-2.5-pro": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-pro",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:llama-4-maverick": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-4-maverick",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:llama-4-scout": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-4-scout",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:qwen3-235b": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3.5-397b-a17b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "openrouter:qwen3-32b": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3.5-27b",
        "temperature": 0,
        "max_tokens": 1024,
    },
    # Keep direct Google API as fallback (rate limited)
    "google:gemini-2.5-flash": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
        "temperature": 0,
        "max_tokens": 1024,
    },
    "google:gemini-2.5-pro": {
        "provider": "google",
        "model_id": "gemini-2.5-pro",
        "temperature": 0,
        "max_tokens": 1024,
    },
}

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
TASKS_DIR = PROJECT_ROOT / "data" / "tasks"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ────────────────────────────────────────────────────

WMF_LOAD_LEVELS = [3, 5, 7, 10, 14]          # number of target items (N)
WMF_OPERATION_DEPTHS = [2, 4, 6, 8, 12]       # update sequence lengths (K)

MCC_PROBLEMS_PER_DOMAIN = {
    "mmlu": 200,
    "math": 100,
    "ood_synthetic": 50,
}
CONFIDENCE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


EMC_EVENT_COUNTS = [5, 10, 15, 20]            # temporal ordering
EMC_SOURCE_COUNTS = [2, 3, 5]                 # source attribution
EMC_EPISODE_COUNTS = [1, 2, 3, 5]            # interference

# ── Utility: call any model ──────────────────────────────────────────────────

def call_model(
    model_name: str,
    prompt: str,
    system: str = "",
    history: list[dict] | None = None,
) -> str:
    """
    Call the specified model with a prompt and return the text response.

    Args:
        model_name: Key into MODELS registry.
        prompt: The new user message to send.
        system: System prompt (used for all providers).
        history: Prior conversation turns as list of {"role": ..., "content": ...} dicts.
                 The new user `prompt` is appended to this history before sending.
                 If None, starts a fresh single-turn conversation.
    """
    cfg = MODELS[model_name]
    provider = cfg["provider"]

    # Build the full message list: [system?] + history + [new user turn]
    def _build_messages() -> list[dict]:
        msgs = []
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": prompt})
        return msgs

    if provider in ("openai", "ollama", "openrouter"):
        import openai
        if provider == "openai":
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif provider == "openrouter":
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            client = openai.OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1/", api_key="ollama")
        msgs = _build_messages()
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        response = client.chat.completions.create(
            model=cfg["model_id"],
            messages=msgs,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        content = response.choices[0].message.content
        return content.strip() if content else None

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=cfg["model_id"],
            max_tokens=cfg["max_tokens"],
            system=system or "You are a helpful assistant.",
            messages=_build_messages(),
        )
        return response.content[0].text.strip()

    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(cfg["model_id"])
        # Google GenAI doesn't support multi-turn history in the simple API,
        # so we flatten history into the prompt.
        history_text = ""
        if history:
            for turn in history:
                role = "User" if turn["role"] == "user" else "Assistant"
                history_text += f"{role}: {turn['content']}\n"
        full_prompt = ""
        if system:
            full_prompt += system + "\n\n"
        if history_text:
            full_prompt += history_text
        full_prompt += f"User: {prompt}\nAssistant:"
        response = model.generate_content(full_prompt)
        return response.text.strip()

    elif provider == "together":
        from together import Together
        client = Together(api_key=os.environ["TOGETHER_API_KEY"])
        msgs = _build_messages()
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        response = client.chat.completions.create(
            model=cfg["model_id"],
            messages=msgs,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        content = response.choices[0].message.content
        return content.strip() if content else None

    raise ValueError(f"Unknown provider: {provider}")
