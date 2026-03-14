"""Translation Tool: translate text using an LLM."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class TranslationTool(BaseTool):
    """Translate text between languages using an LLM.

    Usage::

        tool = TranslationTool(llm=llm)
        result = await tool.run(text="Hello world!", target_language="Spanish")
    """

    name = "translate"
    description = (
        "Translate text from one language to another. "
        "Input: the text and the target language. "
        "Returns: the translated text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to translate",
            },
            "target_language": {
                "type": "string",
                "description": "The language to translate to (e.g., 'Spanish', 'French', 'Japanese')",
            },
            "source_language": {
                "type": "string",
                "description": "The source language (optional, auto-detected if omitted)",
            },
        },
        "required": ["text", "target_language"],
    }

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def run(
        self,
        text: str = "",
        target_language: str = "",
        source_language: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        input_text = text or kwargs.get("input", "")
        if not input_text:
            return ToolResult(output="", error="No text provided to translate.")
        if not target_language:
            return ToolResult(output="", error="No target language specified.")

        if source_language:
            prompt = (
                f"Translate the following text from {source_language} to {target_language}. "
                f"Return only the translated text, nothing else.\n\n"
                f"Text:\n{input_text}"
            )
        else:
            prompt = (
                f"Translate the following text to {target_language}. "
                f"Return only the translated text, nothing else.\n\n"
                f"Text:\n{input_text}"
            )

        try:
            result: str = await self._llm.generate(prompt)
            return ToolResult(output=result.strip())
        except Exception as e:
            return ToolResult(output="", error=f"Translation failed: {e}")
