"""Answer generation. Offline extractive stub by default; LLM when a key is set."""
from __future__ import annotations

from .vectorstore import Hit


def _context_block(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i+1}] {h.chunk.text}" for i, h in enumerate(hits))


class ExtractiveGenerator:
    """No-LLM fallback: returns the top retrieved passages as the 'answer'.

    Useful so the API and pipeline are runnable offline. It is deliberately dumb --
    the point of this repo is the eval harness, not a clever offline generator.
    """

    def generate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return "No relevant context found."
        return hits[0].chunk.text


class LLMGenerator:
    """Grounded generation via OpenAI (optional dependency)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, hits: list[Hit]) -> str:
        context = _context_block(hits)
        prompt = (
            "Answer the question using only the context. Cite passages like [1], [2]. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


def build_generator(openai_api_key: str | None):
    if openai_api_key:
        return LLMGenerator(openai_api_key)
    return ExtractiveGenerator()
