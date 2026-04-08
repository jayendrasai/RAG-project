import logging
import json
from typing import List, Dict, Optional

from openai import OpenAI
from groq import Groq
from backend.config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a document analysis assistant. Answer questions using ONLY the provided context.

Rules:
- Cite specific page numbers when referencing content (e.g., "On page 3...")
- If the answer comes from a chart, table, or image, explicitly say so
- Reference the document filename when citing sources
- If the context doesn't contain enough info, say so clearly
- Be concise and direct"""




# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    chunks: List[Dict],
    chat_history: List[Dict] = None,
    images: List[str] = None,
) -> Dict:
    """
    Call the primary LLM provider; if it fails, automatically fall back to
    the other provider. Controlled by LLM_PROVIDER env var (default: groq).
    """
    primary = settings.llm_provider.lower()
    fallback = "openai" if primary == "groq" else "groq"
    order = [primary, fallback]

    # If neither key is set, return mock
    if not settings.openai_api_key and not settings.groq_api_key:
        return _mock_response(query, chunks)

    # Build context and messages
    context_block = _build_context(chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context_block}\n\nQuestion: {query}",
    })

    # Try providers in order
    for provider_name in order:
        call_fn, get_key = _PROVIDERS[provider_name]
        api_key = get_key()

        if not api_key:
            log.info(f"Skipping {provider_name} — no API key configured.")
            continue

        try:
            log.info(f"Calling {provider_name}...")
            answer = call_fn(messages, images)
            sources = _build_sources(chunks)
            return {"answer": answer, "sources": sources}

        except Exception as e:
            log.warning(f"{provider_name} failed: {e}")
            log.info(f"Falling back to next provider...")

    # Both failed
    return {
        "answer": "Sorry, all LLM providers failed. Please check your API keys and quotas.",
        "sources": [],
    }

