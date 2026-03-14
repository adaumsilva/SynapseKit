"""Text Summarization Tool: summarize text using an LLM."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class SummarizationTool(BaseTool):
    """Summarize text using an LLM.

    Usage::

        tool = SummarizationTool(llm=llm)
        result = await tool.run(text="Long article text here...")
    """

    name = "summarize"
    description = (
        "Summarize a piece of text. "
        "Input: the text to summarize, and optionally a max_sentences count. "
        "Returns: a concise summary of the text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to summarize",
            },
            "max_sentences": {
                "type": "integer",
                "description": "Maximum number of sentences in the summary (default: 3)",
                "default": 3,
            },
            "style": {
                "type": "string",
                "description": "Summary style: 'concise', 'bullet_points', or 'detailed' (default: 'concise')",
                "default": "concise",
            },
        },
        "required": ["text"],
    }

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def run(
        self, text: str = "", max_sentences: int = 3, style: str = "concise", **kwargs: Any
    ) -> ToolResult:
        input_text = text or kwargs.get("input", "")
        if not input_text:
            return ToolResult(output="", error="No text provided to summarize.")

        if style == "bullet_points":
            prompt = (
                f"Summarize the following text as {max_sentences} bullet points. "
                f"Return only the bullet points, one per line starting with '- '.\n\n"
                f"Text:\n{input_text}"
            )
        elif style == "detailed":
            prompt = (
                f"Provide a detailed summary of the following text in {max_sentences} sentences. "
                f"Capture all key points and nuances.\n\n"
                f"Text:\n{input_text}"
            )
        else:
            prompt = (
                f"Summarize the following text in {max_sentences} sentences or fewer. "
                f"Be concise and capture the main points.\n\n"
                f"Text:\n{input_text}"
            )

        try:
            result: str = await self._llm.generate(prompt)
            return ToolResult(output=result.strip())
        except Exception as e:
            return ToolResult(output="", error=f"Summarization failed: {e}")
